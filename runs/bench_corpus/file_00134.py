from django.db.models import Q
from common_py.models import contract, contract_version
from common_py.utils import getNowDataTime

#계약서의 버전리스트와, 최신버전 한건을 습득한다.
def getContractVersion(contractId):
    try:
        # 버전 역순으로 검색, 첫번째것이 최신버전임.
        ctVersionResult = contract_version.objects.filter(contract_id = contractId).order_by('version').reverse()
        contractVersionList = list(ctVersionResult)
        # 데이터가 안들어가있으면 html에서 사용하기 곤란하기때문에 공백문자로 초기화해준다.
        lastContractVersion = {
            'contract_id':'',
            'contract_name':'',
            'contract_content':'',
            'cv_eos_reg_date':'',
            'cv_eos_hash':'',
            'version':'',
            'reg_date':'',
        }
        for versionItem in contractVersionList:
            eosHash = contractVersionList[0].cv_eos_hash
            cv_eos_reg_date = contractVersionList[0].cv_eos_reg_date
            if eosHash is None:
                eosHash = ''
            if cv_eos_reg_date is None:
                cv_eos_reg_date = ''
            lastContractVersion = {
                'contract_id':contractVersionList[0].contract_id,
                'contract_name':contractVersionList[0].contract_name,
                'contract_content':contractVersionList[0].contract_content,
                'cv_eos_reg_date':cv_eos_reg_date,
                'cv_eos_hash':eosHash,
                'version':contractVersionList[0].version,
                'reg_date':contractVersionList[0].reg_date,
            }
            break

    except Exception as ex:
        msg = "Exception.getContractVersion(contractId:" + str(contractId) + ") ex:" + str(ex)
        print(msg)
        return {'result_code': -9999,'msg':'Exception : ' + str(ex)}
    
    result = {'result_code': 0,'msg':'success', 'contractVersionList':contractVersionList, 'lastContractVersion':lastContractVersion}

    return result

# 계약서 버전 신규 등록 및 버전업.
def setContractVersionUp(contractId, contractTitle, contractContent):
    try:
        # 버전 역순으로 검색, 첫번째것이 최신버전임.
        ctVersionResult = contract_version.objects.filter(contract_id = contractId).order_by('version').reverse()[:1]
        nowVersion = 0
        contractVersionList = None
        if ctVersionResult is None or len(list(ctVersionResult)) == 0:
            nowVersion = 0
        else:
            contractVersionList = list(ctVersionResult)
            nowVersionItem = contractVersionList[0]
            nowVersion = nowVersionItem.version
            if nowVersion is None:
                nowVersion = 0
        newVersion = nowVersion + 1

        # 계약서 용량하고 ifps해쉬는 EOS등록할때 정해지므로 버전업시에는 0과 None으로 넣는다.
        contractSize = 0
        contractIPFSHash = None

        #신규등록, 및 버전업때는 EOS해쉬는 초기화된다.
        nowDate = getNowDataTime()
        newVerObj = contract_version(contract_id = contract.objects.get(contract_id = contractId), contract_name=contractTitle, contract_content=contractContent, \
            version = newVersion, cv_eos_reg_date = None, cv_eos_hash=  None ,cv_contract_size = contractSize, \
            cv_contract_hash=contractIPFSHash, reg_date = nowDate, update_date = nowDate)
        newVerObj.save()

    except Exception as ex:
        msg = "Exception.setContractVersionUp(contractId:" + str(contractId) + ") ex:" + str(ex)
        print(msg)
        return {'result_code': -9999,'msg':'Exception : ' + str(ex)}
    
    result = {'result_code': 0,'msg':'success'}

    return result