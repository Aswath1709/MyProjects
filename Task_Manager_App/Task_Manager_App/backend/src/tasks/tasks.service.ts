import {
  Injectable,
  OnModuleInit,
  NotFoundException,
  ConflictException,
  InternalServerErrorException,
} from '@nestjs/common';
import Nano from 'nano';
import { CreateTaskDto } from './dto/create-task.dto';
import { Task } from './interfaces/task.interface';
import { UpdateTaskDto } from './dto/update-task.dto';
import { ElasticsearchService } from '../elasticsearch/search-service';

declare const emit: any;

@Injectable()
export class TasksService implements OnModuleInit {
  private db: Nano.DocumentScope<Task>;
  private nanoClient: Nano.ServerScope;

  constructor(
    private readonly elasticsearchService: ElasticsearchService,
  ) {}

  async onModuleInit() {
    const couchUrl = process.env.COUCHDB_URL || 'http://admin:password@localhost:5984';
    console.log(`[TasksService] Using CouchDB URL: ${couchUrl}`);
    this.nanoClient = Nano(couchUrl);
    const dbName = 'tasks';

    try {
      await this.nanoClient.db.get(dbName);
      console.log(`CouchDB database '${dbName}' already exists.`);
    } catch (error: any) {
      if (error.statusCode === 404) {
        try {
          await this.nanoClient.db.create(dbName);
          console.log(`CouchDB database '${dbName}' created successfully.`);
        } catch (createError: any) {
          console.error(`Failed to create CouchDB database '${dbName}':`, createError);
          throw new InternalServerErrorException(`Failed to initialize tasks database.`);
        }
      } else {
        console.error('Failed to initialize CouchDB connection or check database:', error);
        throw new InternalServerErrorException('Failed to connect to CouchDB.');
      }
    }

    this.db = this.nanoClient.db.use(dbName);

    const designDocId = '_design/tasks';
    try {
      await this.db.get(designDocId);
      console.log(`CouchDB design document '${designDocId}' already exists.`);
    } catch (error: any) {
      if (error.statusCode === 404) {
        try {
          await this.db.insert({
            _id: designDocId,
            views: {
              all_tasks_by_user: {
                map: function (doc) {
                  if (doc.userId) {
                    emit(doc.userId, null);
                  }
                }.toString(),
              },
              tasks_by_user_and_status: {
                map: function (doc) {
                  if (doc.userId && doc.status) {
                    emit([doc.userId, doc.status], null);
                  }
                }.toString(),
              },
              tasks_by_user_and_title: {
                map: function (doc) {
                  if (doc.userId && doc.title) {
                    emit([doc.userId, doc.title], doc.description);
                  }
                }.toString(),
              },
            },
          });
          console.log(`CouchDB design document and user-specific views created.`);
        } catch (designDocError: any) {
          console.error(`Failed to create CouchDB design document '${designDocId}':`, designDocError);
          throw new InternalServerErrorException(`Failed to set up tasks database views.`);
        }
      } else {
        console.error(`Error checking design document '${designDocId}':`, error);
        throw new InternalServerErrorException(`Failed to access design document.`);
      }
    }

    console.log(`Connected to CouchDB database: ${dbName}`);
  }

  async create(createTaskDto: CreateTaskDto, userId: string): Promise<Task> {
    const newTask: Task = {
      ...createTaskDto,
      description: createTaskDto.description ?? '',
      userId: userId,
      createdAt: new Date().toISOString(),
      status: createTaskDto.status || 'Pending',
    };

    try {
      const response = await this.db.insert(newTask);
      const createdTaskWithId: Task = { _id: response.id, _rev: response.rev, ...newTask };
      await this.elasticsearchService.indexDocument(createdTaskWithId._id!, createdTaskWithId);
      return createdTaskWithId;
    } catch (error: any) {
      console.error('Error creating task in DB:', error);
      throw new InternalServerErrorException('Failed to create task.');
    }
  }

  async findAll(userId: string): Promise<Task[]> {
    try {
      const result = await this.db.view('tasks', 'all_tasks_by_user', { key: userId, include_docs: true });
      if (!result || !result.rows) {
        return [];
      }
      return result.rows
        .map(row => row.doc)
        .filter(doc => doc && !doc._id?.startsWith('_design/')) as Task[];
    } catch (error: any) {
      console.error(`Error finding all tasks for user ${userId}:`, error);
      throw new InternalServerErrorException('Failed to retrieve tasks.');
    }
  }

  async findOne(id: string, userId: string): Promise<Task> {
    try {
      const task = await this.db.get(id) as Task;
      if (!task || task.userId !== userId) {
        throw new NotFoundException(`Task with ID ${id} not found or does not belong to user.`);
      }
      return task;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new NotFoundException(`Task with ID "${id}" not found.`);
      }
      console.error(`Error fetching task with ID ${id} for user ${userId}:`, error);
      throw new InternalServerErrorException('Failed to retrieve task.');
    }
  }

  async update(id: string, updateTaskDto: UpdateTaskDto, userId: string): Promise<Task> {
    try {
      const existingTask = await this.db.get(id) as Task;

      if (!existingTask || existingTask.userId !== userId) {
        throw new NotFoundException(`Task with ID ${id} not found or does not belong to user.`);
      }

      const updatedTask = {
        ...existingTask,
        ...updateTaskDto,
        _id: existingTask._id,
        _rev: existingTask._rev,
        userId: existingTask.userId, // preserve original owner
      } as Task;

      const response = await this.db.insert(updatedTask);
      await this.elasticsearchService.indexDocument(updatedTask._id!, updatedTask);

      return { ...updatedTask, _rev: response.rev };
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new NotFoundException(`Task with ID "${id}" not found for update.`);
      }
      console.error(`Error updating task with ID ${id} for user ${userId}:`, error);
      throw new InternalServerErrorException('Failed to update task.');
    }
  }

  async remove(id: string, userId: string) {
    try {
      const existingTask = await this.db.get(id) as Task;

      if (!existingTask || existingTask.userId !== userId) {
        throw new NotFoundException(`Task with ID ${id} not found or does not belong to user.`);
      }

      const response = await this.db.destroy(existingTask._id!, existingTask._rev!);
      await this.elasticsearchService.deleteDocument(id);

      return response;
    } catch (error: any) {
      if (error.statusCode === 404) {
        throw new NotFoundException(`Task with ID ${id} not found.`);
      }
      if (error.statusCode === 409) {
        throw new ConflictException(`Conflict: Revision mismatch for task with ID ${id}.`);
      }
      throw new InternalServerErrorException(`Failed to delete task with ID ${id}.`);
    }
  }

  async searchTasks(query: string, userId: string): Promise<Task[]> {
    if (!query || query.trim() === '') {
      return this.findAll(userId);
    }
    const results = await this.elasticsearchService.searchDocuments(query, userId);
    return results as Task[];
  }
}
