import { jest } from '@jest/globals';
export const elasticsearchServiceMock = {
  index: jest.fn(),
  search: jest.fn(),
  delete: jest.fn(),
};