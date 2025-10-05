import { Test, TestingModule } from '@nestjs/testing';
import { TasksController } from './tasks.controller';
import { TasksService } from './tasks.service';
import { ElasticsearchService } from '../elasticsearch/search-service';

describe('TasksController', () => {
  let controller: TasksController;

  const mockElasticService = {
    indexDocument: jest.fn(),
    deleteDocument: jest.fn(),
    searchDocuments: jest.fn().mockResolvedValue([]),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TasksController],
      providers: [
        TasksService,
        {
          provide: ElasticsearchService,
          useValue: mockElasticService,
        },
      ],
    }).compile();

    controller = module.get<TasksController>(TasksController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});