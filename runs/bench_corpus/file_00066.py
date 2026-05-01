import sys
from PyQt5 import QtCore, QtWidgets, uic
from tests import BiomioTest
import time
from biomio.protocol.settings import settings

form_class = uic.loadUiType("untitled.ui")[0]


class MyWindowClass(QtWidgets.QMainWindow, form_class):
    testSignal = QtCore.pyqtSignal()
    rpcSendSignal = QtCore.pyqtSignal()
    startSignal = QtCore.pyqtSignal()

    testSignal_probe = QtCore.pyqtSignal()
    touchid_success_toggled = QtCore.pyqtSignal(bool)
    startSignal_probe = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton_rpc.clicked.connect(self.pushButtonRpcClicked)
        self.pushButton_start.clicked.connect(self.pushButtonStartClicked)

        self.pushButton_probe_start.clicked.connect(self.pushButtonProbeStartClicked)
        self.checkBox_touchid_success.toggled.connect(self.probe_checkbox_toggled)
        self.pushButton_probe.clicked.connect(self.pushButtonProbeClicked)

    @QtCore.pyqtSlot()
    def pushButtonClicked(self):
        self.testSignal.emit()
        print "bye clicked"

    @QtCore.pyqtSlot()
    def pushButtonRpcClicked(self):
        self.rpcSendSignal.emit()
        print "rpc clicked"

    @QtCore.pyqtSlot()
    def pushButtonStartClicked(self):
        self.startSignal.emit()
        print "start clicked"

    @QtCore.pyqtSlot()
    def showRpcFinished(self):
        QtWidgets.QMessageBox.information(None, 'extension', 'rpc finished!')

    QtCore.pyqtSlot()
    def pushButtonProbeClicked(self):
        self.testSignal_probe.emit()
        print "bye clicked [probe]"

    @QtCore.pyqtSlot(bool)
    def probe_checkbox_toggled(self, toggled):
        self.touchid_success_toggled.emit(toggled)
        print "touch id success = %s [probe]" % toggled

    @QtCore.pyqtSlot()
    def pushButtonProbeStartClicked(self):
        self.startSignal_probe.emit()
        print "start clicked [probe]"


class ExtensionSession(QtCore.QThread):

    rpcFinishedSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        self.register = False
        self.disconnect = False
        self.rpc_req = False
        self.reg_key = None
        self.reg_id = None
        self.app_id = None

    @QtCore.pyqtSlot()
    def slot_bye(self):
        self.disconnect = True

    @QtCore.pyqtSlot()
    def slot_rpc(self):
        print 'rpc_req', self.rpc_req
        self.rpc_req = True

    def run(self):
        test_obj = BiomioTest()
        if self.register:
            test_obj.setup_test_for_for_new_id()
            test_obj._builder.set_header(appType='extension')

            # CLIENT HELLO ->
            # SERVER HELLO <-
            message = test_obj.create_next_message(oid='clientHello', secret='secret')
            response = test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False, wait_for_response=True)

            # ACK ->
            message = test_obj.create_next_message(oid='ack')
            test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False,
                              wait_for_response=False)
        else:
            BiomioTest._registered_key = str(self.reg_key)
            BiomioTest._registered_user_id = str(self.reg_id)
            test_obj.setup_test_with_handshake(is_registration_required=False)

        message_timeout = settings.connection_timeout / 2  # Send a message every 3 seconds
        max_message_count = 10
        rpc_responce_message = None
        while(True):
            response = None
            try:
                response = test_obj.read_message(websocket=test_obj.get_curr_connection())
            except Exception, e:
                pass

            # RPC RESP <-
            if response and response.msg and str(response.msg.oid) == 'rpcResp':
                if response.msg.rpcStatus == 'complete':
                    print 'finished!'
                    self.rpcFinishedSignal.emit()

            if self.disconnect:
                probe_msg = test_obj.create_next_message(oid='bye')
                response = test_obj.send_message(websocket=test_obj.get_curr_connection(), message=probe_msg,
                                                 close_connection=False, wait_for_response=True)
                self.disconnect = False
                break
            else:
                if self.rpc_req:
                    message = test_obj.create_next_message(oid='rpcReq', namespace='pgp_extension_plugin', call='test_func_with_auth',
                        data={'keys': ['val1', 'val2'], 'values': ['1', '2']})
                    print "rpc request"
                    test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False,
                        wait_for_response=True)
                    print "send!"
                    self.rpc_req = False

                message = test_obj.create_next_message(oid='nop')
                message.header.token = test_obj.session_refresh_token
                try:
                    # NOP ->
                    # NOP <-
                    response = test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message,
                                                 close_connection=False, wait_for_response=True)
                                # RPC RESP <-
                    if response and response.msg and str(response.msg.oid) == 'rpcResp':
                        if response.msg.rpcStatus == 'complete':
                            print 'finished!'
                            self.rpcFinishedSignal.emit()

                    ok_(str(response.msg.oid) == 'nop', msg='No responce on nop message')
                except Exception, e:
                    pass


class ProbeSession(QtCore.QThread):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        self.register = False
        self.disconnect = False
        self.probe_result = True
        self.reg_key = None
        self.reg_id = None
        self.app_id = None

    @QtCore.pyqtSlot()
    def slot_bye(self):
        self.disconnect = True

    @QtCore.pyqtSlot(bool)
    def slot_set_probe_result(self, result):
        self.probe_result = result

    def run(self):
        test_obj = BiomioTest()
        if self.register:
            test_obj.setup_test_for_for_new_id()
            test_obj._builder.set_header(appType='probe')

            # CLIENT HELLO ->
            message = test_obj.create_next_message(oid='clientHello', secret='secret')
            test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False, wait_for_response=False)

            # RESOURCES ->
            message = test_obj.create_next_message(oid='resources', data=[{"rType": "video", "rProperties": "1500x1000"},
                {"rType": "fp-scanner", "rProperties": "true"}, {"rType": "mic", "rProperties": "true"}])

            # SERVER HELLO <-
            response = test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False, wait_for_response=True)

            # ACK ->
            message = test_obj.create_next_message(oid='ack')
            test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message, close_connection=False,
                              wait_for_response=False)
        else:
            BiomioTest._registered_key = str(self.reg_key)
            BiomioTest._registered_user_id = str(self.reg_id)
            test_obj.setup_test_with_handshake(is_registration_required=False)

        message_timeout = settings.connection_timeout / 2  # Send a message every 3 seconds
        max_message_count = 10
        rpc_responce_message = None
        while(True):
            try:
                message = test_obj.read_message(websocket=test_obj.get_curr_connection())

                # TRY <-
                if message and message.msg and str(message.msg.oid) == 'try':
                    print "!!!!!!!!!!!!!!!!!!!!!!!!"
                    # PROBE ->
                    if self.probe_result:
                        probe_msg = test_obj.create_next_message(oid='probe', probeId=0,
                                                             probeData={"oid": "touchIdSamples", "samples": ["True"]})
                    else:
                        probe_msg = test_obj.create_next_message(oid='probe', probeId=0,
                                                             probeData={"oid": "touchIdSamples", "samples": ["False"]})

                    test_obj.send_message(websocket=test_obj.get_curr_connection(), message=probe_msg, close_connection=False,
                        wait_for_response=False)
                    print "sent probes"
                    continue
            except Exception, e:
                print e
                pass

            if self.disconnect:
                print "bye"
                probe_msg = test_obj.create_next_message(oid='bye')
                test_obj.send_message(websocket=test_obj.get_curr_connection(), message=probe_msg,
                                                 close_connection=False, wait_for_response=True)
                self.disconnect = False
                break
            else:

                message = test_obj.create_next_message(oid='nop')
                message.header.token = test_obj.session_refresh_token
                try:
                    # NOP ->
                    # NOP <-
                    response = test_obj.send_message(websocket=test_obj.get_curr_connection(), message=message,
                                                 close_connection=False, wait_for_response=True)
                    ok_(str(response.msg.oid) == 'nop', msg='No responce on nop message')
                except Exception, e:
                    pass


def main():
    app = QtWidgets.QApplication(sys.argv)

    win = MyWindowClass(None)
    win.show()

    extentsion = ExtensionSession(None)
    extentsion.register = True
    win.testSignal.connect(extentsion.slot_bye)
    win.rpcSendSignal.connect(extentsion.slot_rpc)
    win.startSignal.connect(extentsion.start)
    extentsion.rpcFinishedSignal.connect(win.showRpcFinished)

    probe = ProbeSession(None)
    probe.register = True
    win.testSignal_probe.connect(probe.slot_bye)
    win.touchid_success_toggled.connect(probe.slot_set_probe_result)
    win.startSignal_probe.connect(probe.start)

    app.exec_()

if __name__ == '__main__':
    main()
