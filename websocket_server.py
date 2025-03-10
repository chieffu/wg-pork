# websocket_server.py
import asyncio
import websockets

class WebSocketServer:
    def __init__(self,logger=None, host='0.0.0.0', websocket_port=8765, udp_port=5005, loop=None):
        self.logger=logger
        self.host = host
        self.websocket_port = websocket_port
        self.udp_port = udp_port
        self.clients = set()
        self.received_messages = {}
        self.websocket_server = None
        self.udp_listener = None
        self.loop = loop or asyncio.get_event_loop()

    async def ws_handler(self, websocket):
        self.clients.add(websocket)
        self.logger.info(f"New WebSocket connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                # 处理接收到的消息
                self.logger.info(f"Received message from {websocket.remote_address}: {message}")
                # 可以在这里添加消息处理逻辑
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            self.logger.info(f"WebSocket connection closed from {websocket.remote_address}")

    async def udp_server(self):
        loop = asyncio.get_event_loop()
        self.logger.info(f"Starting UDP server on port {self.udp_port}")
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UdpProtocol(self),
            local_addr=(self.host, self.udp_port))

        try:
            while True:
                await asyncio.sleep(3600)  # Keep the task alive
        except asyncio.CancelledError:
            transport.close()
            self.logger.info("UDP server stopped")

    def validate_message(self, message):
        # 示例判断逻辑：检查消息是否包含特定字符串
        parts = message.rsplit(',', 1)
        if len(parts) != 2:
            self.logger.warning(f"Invalid message format: {message}")
            return False
        key, timestamp = parts
        if key in self.received_messages:
            if abs(float(timestamp) - self.received_messages[key]) < 15:
                self.logger.info(f"Duplicate message received: {message}")
                return False
        info = key.split(",")
        if len(info) != 2:
            self.logger.warning(f"Invalid key format: {key}")
            return False
        self.received_messages[key] = float(timestamp)
        return True

    async def broadcast_message(self, message):
        if not self.validate_message(message):
            return
        await self._broadcast_message(message)

    async def _broadcast_message(self, message):
        self.logger.info(f"Broadcasting message to {len(self.clients)} clients: {message}")
        for websocket in self.clients.copy():
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                self.clients.remove(websocket)
                self.logger.info(f"Removed closed connection: {websocket.remote_address}")

    async def start(self):
        self.websocket_server = await websockets.serve(self.ws_handler, self.host, self.websocket_port)
        self.logger.info(f"WebSocket server started on port {self.websocket_port}")
        self.udp_listener = asyncio.create_task(self.udp_server())
        await asyncio.gather(self.websocket_server.serve_forever(), self.udp_listener)

    async def stop(self):
        self.websocket_server.close()
        await self.websocket_server.wait_closed()
        self.udp_listener.cancel()
        await self.udp_listener


class UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, websocket_server):
        self.websocket_server = websocket_server

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode()
        self.logger.info(f"Received UDP message from {addr}: {message}")
        asyncio.create_task(self.websocket_server.broadcast_message(message))


if __name__ == "__main__":
    server = WebSocketServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        server.logger.info("Server stopped.")
        asyncio.run(server.stop())
