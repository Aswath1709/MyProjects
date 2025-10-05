import { Test, TestingModule } from '@nestjs/testing';
import { TasksService } from './tasks.service';
import { ElasticsearchService } from '../elasticsearch/search-service';

describe('TasksService', () => {
  let service: TasksService;

  const mockElasticService = {
    indexDocument: jest.fn(),
    deleteDocument: jest.fn(),
    searchDocuments: jest.fn().mockResolvedValue([]),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        TasksService,
        {
          provide: ElasticsearchService,
          useValue: mockElasticService,
        },
      ],
    }).compile();

    service = module.get<TasksService>(TasksService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});