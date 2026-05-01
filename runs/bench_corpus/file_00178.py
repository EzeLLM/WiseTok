"""Server starts serving from here
"""
import asyncio
import signal
from server_file import Server

signal.signal(signal.SIGINT, signal.SIG_DFL)
user_dictionary = {}
async def handle_echo(reader, writer):
    """Server is created and waits for client to get connected to server.
    This server can make connection to multiple clients """
    # data = await reader.read(100)
    message = "connection successful"
    info = writer.get_extra_info('peername')
    user_dictionary[info[1]] = Server()
    print(f"{message!r} from {info!r}")
    while True:
        msg = await reader.read(10000)
        message = msg.decode().strip()
        if message == 'quit':
            user_dictionary[info[1]].remove_user()
            break
        print(f"Received {message} from {info}")
        res_msg = user_dictionary[info[1]].split_details(message)
        print(res_msg)
        if res_msg != '' or res_msg != 'None':
            writer.write(res_msg.encode())
        else:
            res_msg = '.'
            writer.write(res_msg.encode())
        await writer.drain()
    print("close the connection")
    writer.close()
    # print(f"Send: {message!r}")
    # writer.write(data)
    # await writer.drain()

    # print("Close the connection")
    # writer.close()

async def main():
    """The entire program starts from this main function"""
    server = await asyncio.start_server(
        handle_echo, '127.0.0.1', 8888)

    info = server.sockets[0].getsockname()
    print(f'Serving on {info}')
    async with server:
        await server.serve_forever()

asyncio.run(main())
