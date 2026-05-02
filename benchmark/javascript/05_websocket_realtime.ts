import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { v4 as uuid } from 'uuid';

interface Message {
  type: string;
  payload: unknown;
  timestamp: number;
  id: string;
}

interface ClientMessage extends Message {
  room?: string;
  userId?: string;
}

interface ServerMessage extends Message {
  from?: string;
  room?: string;
}

interface ClientConnection {
  id: string;
  socket: WebSocket;
  userId?: string;
  rooms: Set<string>;
  lastHeartbeat: number;
}

export class RealtimeServer extends EventEmitter {
  private wss: WebSocket.Server;
  private clients: Map<string, ClientConnection> = new Map();
  private rooms: Map<string, Set<string>> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectionTimeout: number = 30000;

  constructor(port: number = 8080) {
    super();
    this.wss = new WebSocket.Server({ port });
    this.setupServer();
  }

  private setupServer(): void {
    this.wss.on('connection', (socket: WebSocket) => {
      const clientId = uuid();
      const client: ClientConnection = {
        id: clientId,
        socket,
        rooms: new Set(),
        lastHeartbeat: Date.now()
      };

      this.clients.set(clientId, client);
      console.log(`Client connected: ${clientId}`);

      socket.on('message', (data: WebSocket.Data) => this.handleMessage(clientId, data));
      socket.on('close', () => this.handleDisconnect(clientId));
      socket.on('error', (err) => this.handleError(clientId, err));

      this.sendToClient(clientId, {
        type: 'welcome',
        payload: { clientId, serverTime: Date.now() },
        timestamp: Date.now(),
        id: uuid()
      });
    });

    this.startHeartbeat();
  }

  private handleMessage(clientId: string, data: WebSocket.Data): void {
    try {
      const message: ClientMessage = JSON.parse(data.toString());
      const client = this.clients.get(clientId);

      if (!client) return;

      client.lastHeartbeat = Date.now();

      switch (message.type) {
        case 'identify':
          this.handleIdentify(clientId, message);
          break;

        case 'join_room':
          this.handleJoinRoom(clientId, message.room!);
          break;

        case 'leave_room':
          this.handleLeaveRoom(clientId, message.room!);
          break;

        case 'broadcast':
          this.handleBroadcast(clientId, message);
          break;

        case 'direct':
          this.handleDirectMessage(clientId, message);
          break;

        case 'pong':
          this.handlePong(clientId);
          break;

        default:
          this.emit('message', { clientId, message });
      }
    } catch (err) {
      console.error(`Message parse error for ${clientId}:`, err);
    }
  }

  private handleIdentify(clientId: string, message: ClientMessage): void {
    const client = this.clients.get(clientId);
    if (client && message.payload) {
      const payload = message.payload as { userId: string };
      client.userId = payload.userId;
      this.sendToClient(clientId, {
        type: 'identified',
        payload: { clientId, userId: client.userId },
        timestamp: Date.now(),
        id: uuid()
      });
    }
  }

  private handleJoinRoom(clientId: string, room: string): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.rooms.add(room);
    if (!this.rooms.has(room)) {
      this.rooms.set(room, new Set());
    }
    this.rooms.get(room)!.add(clientId);

    this.broadcastToRoom(room, {
      type: 'user_joined',
      payload: { userId: client.userId, clientId },
      timestamp: Date.now(),
      id: uuid(),
      room
    }, clientId);

    console.log(`Client ${clientId} joined room ${room}`);
  }

  private handleLeaveRoom(clientId: string, room: string): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.rooms.delete(room);
    const roomClients = this.rooms.get(room);
    if (roomClients) {
      roomClients.delete(clientId);
      if (roomClients.size === 0) {
        this.rooms.delete(room);
      } else {
        this.broadcastToRoom(room, {
          type: 'user_left',
          payload: { userId: client.userId, clientId },
          timestamp: Date.now(),
          id: uuid(),
          room
        });
      }
    }

    console.log(`Client ${clientId} left room ${room}`);
  }

  private handleBroadcast(clientId: string, message: ClientMessage): void {
    const client = this.clients.get(clientId);
    if (!client || !message.room) return;

    this.broadcastToRoom(message.room, {
      type: 'broadcast',
      payload: message.payload,
      from: client.userId || clientId,
      timestamp: Date.now(),
      id: uuid(),
      room: message.room
    });
  }

  private handleDirectMessage(clientId: string, message: ClientMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    const payload = message.payload as { targetClientId: string };
    this.sendToClient(payload.targetClientId, {
      type: 'direct',
      payload: message.payload,
      from: client.userId || clientId,
      timestamp: Date.now(),
      id: uuid()
    });
  }

  private handlePong(clientId: string): void {
    const client = this.clients.get(clientId);
    if (client) {
      client.lastHeartbeat = Date.now();
    }
  }

  private handleDisconnect(clientId: string): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    for (const room of client.rooms) {
      this.handleLeaveRoom(clientId, room);
    }

    this.clients.delete(clientId);
    console.log(`Client disconnected: ${clientId}`);
  }

  private handleError(clientId: string, error: Error): void {
    console.error(`WebSocket error for ${clientId}:`, error);
  }

  private broadcastToRoom(room: string, message: ServerMessage, excludeClientId?: string): void {
    const roomClients = this.rooms.get(room);
    if (!roomClients) return;

    roomClients.forEach(clientId => {
      if (excludeClientId !== clientId) {
        this.sendToClient(clientId, message);
      }
    });
  }

  private sendToClient(clientId: string, message: ServerMessage): void {
    const client = this.clients.get(clientId);
    if (client && client.socket.readyState === WebSocket.OPEN) {
      client.socket.send(JSON.stringify(message));
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      this.clients.forEach((client, clientId) => {
        if (now - client.lastHeartbeat > this.reconnectionTimeout) {
          console.log(`Client ${clientId} heartbeat timeout, closing connection`);
          client.socket.close();
        } else {
          this.sendToClient(clientId, {
            type: 'ping',
            payload: { serverTime: now },
            timestamp: now,
            id: uuid()
          });
        }
      });
    }, 30000);
  }

  public stop(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    this.wss.close();
  }

  public getStats() {
    return {
      connectedClients: this.clients.size,
      activeRooms: this.rooms.size,
      rooms: Array.from(this.rooms.entries()).map(([name, clients]) => ({
        name,
        clientCount: clients.size
      }))
    };
  }
}

export default RealtimeServer;
