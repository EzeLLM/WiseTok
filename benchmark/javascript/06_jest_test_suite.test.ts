import { describe, it, beforeEach, afterEach, jest, expect } from '@jest/globals';
import UserService from '../src/UserService';
import DatabaseClient from '../src/DatabaseClient';

jest.mock('../src/DatabaseClient');

describe('UserService', () => {
  let userService: UserService;
  let mockDb: jest.Mocked<typeof DatabaseClient>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockDb = DatabaseClient as jest.Mocked<typeof DatabaseClient>;
    userService = new UserService(mockDb);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('createUser', () => {
    it('should create a user with valid data', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        age: 30
      };

      mockDb.insert.mockResolvedValue({ id: '123', ...userData });

      const result = await userService.createUser(userData);

      expect(result).toEqual({ id: '123', ...userData });
      expect(mockDb.insert).toHaveBeenCalledWith('users', userData);
      expect(mockDb.insert).toHaveBeenCalledTimes(1);
    });

    it('should throw error for invalid email', async () => {
      const invalidData = {
        email: 'not-an-email',
        name: 'Test User'
      };

      await expect(userService.createUser(invalidData)).rejects.toThrow('Invalid email');
    });

    it('should throw error when database fails', async () => {
      const userData = { email: 'test@example.com', name: 'Test User' };
      const dbError = new Error('Database connection failed');

      mockDb.insert.mockRejectedValue(dbError);

      await expect(userService.createUser(userData)).rejects.toThrow('Database connection failed');
    });
  });

  describe('getUser', () => {
    it('should fetch user by ID', async () => {
      const userId = '123';
      const user = { id: userId, email: 'test@example.com', name: 'Test User' };

      mockDb.findById.mockResolvedValue(user);

      const result = await userService.getUser(userId);

      expect(result).toEqual(user);
      expect(mockDb.findById).toHaveBeenCalledWith('users', userId);
    });

    it('should return null for non-existent user', async () => {
      mockDb.findById.mockResolvedValue(null);

      const result = await userService.getUser('non-existent');

      expect(result).toBeNull();
    });
  });

  describe('updateUser', () => {
    it('should update user data', async () => {
      const userId = '123';
      const updates = { name: 'Updated Name', age: 31 };
      const updatedUser = { id: userId, email: 'test@example.com', ...updates };

      mockDb.update.mockResolvedValue(updatedUser);

      const result = await userService.updateUser(userId, updates);

      expect(result).toEqual(updatedUser);
      expect(mockDb.update).toHaveBeenCalledWith('users', userId, updates);
    });

    it('should handle partial updates', async () => {
      const userId = '123';
      mockDb.update.mockResolvedValue({ id: userId, name: 'New Name' });

      await userService.updateUser(userId, { name: 'New Name' });

      expect(mockDb.update).toHaveBeenCalled();
    });
  });

  describe('deleteUser', () => {
    it('should delete user by ID', async () => {
      const userId = '123';
      mockDb.delete.mockResolvedValue(true);

      const result = await userService.deleteUser(userId);

      expect(result).toBe(true);
      expect(mockDb.delete).toHaveBeenCalledWith('users', userId);
    });

    it('should throw error if deletion fails', async () => {
      mockDb.delete.mockRejectedValue(new Error('Deletion failed'));

      await expect(userService.deleteUser('123')).rejects.toThrow('Deletion failed');
    });
  });

  describe('listUsers', () => {
    it('should return paginated users', async () => {
      const users = [
        { id: '1', email: 'user1@example.com', name: 'User 1' },
        { id: '2', email: 'user2@example.com', name: 'User 2' }
      ];

      mockDb.find.mockResolvedValue(users);

      const result = await userService.listUsers({ page: 1, limit: 10 });

      expect(result).toHaveLength(2);
      expect(mockDb.find).toHaveBeenCalled();
    });
  });

  describe('searchUsers', () => {
    it('should search users by query', async () => {
      const query = 'john';
      const results = [
        { id: '1', email: 'john@example.com', name: 'John Doe' }
      ];

      mockDb.search.mockResolvedValue(results);

      const result = await userService.searchUsers(query);

      expect(result).toEqual(results);
      expect(mockDb.search).toHaveBeenCalledWith('users', query);
    });
  });

  describe('snapshot testing', () => {
    it('should match user creation snapshot', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        age: 30,
        role: 'admin'
      };

      mockDb.insert.mockResolvedValue({ id: '123', ...userData });

      const result = await userService.createUser(userData);

      expect(result).toMatchSnapshot();
    });

    it('should match user list snapshot', async () => {
      const users = [
        { id: '1', email: 'user1@example.com', name: 'User 1', role: 'user' },
        { id: '2', email: 'user2@example.com', name: 'User 2', role: 'admin' }
      ];

      mockDb.find.mockResolvedValue(users);

      const result = await userService.listUsers({ page: 1, limit: 10 });

      expect(result).toMatchSnapshot();
    });
  });

  describe('batch operations', () => {
    it('should create multiple users', async () => {
      const users = [
        { email: 'user1@example.com', name: 'User 1' },
        { email: 'user2@example.com', name: 'User 2' }
      ];

      mockDb.batchInsert.mockResolvedValue([
        { id: '1', ...users[0] },
        { id: '2', ...users[1] }
      ]);

      const result = await userService.batchCreateUsers(users);

      expect(result).toHaveLength(2);
      expect(mockDb.batchInsert).toHaveBeenCalledWith('users', users);
    });
  });
});
