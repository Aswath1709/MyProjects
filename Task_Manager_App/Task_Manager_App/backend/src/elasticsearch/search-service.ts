import { Injectable, OnModuleInit, OnModuleDestroy, Logger, InternalServerErrorException } from '@nestjs/common';
import { Client } from '@elastic/elasticsearch';
import { Task } from '../tasks/interfaces/task.interface';

@Injectable()
export class ElasticsearchService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(ElasticsearchService.name);
  private client: Client;
  private readonly indexName = 'tasks';

  async onModuleInit() {
    const node = process.env.ELASTICSEARCH_NODE || 'http://localhost:9200';
    this.logger.log(`Connecting to Elasticsearch at ${node}`);
    this.client = new Client({ node });

    try {
      await this.client.ping();
      this.logger.log('Successfully connected to Elasticsearch.');

      const exists = await this.client.indices.exists({ index: this.indexName });

      if (!exists) {
        await this.client.indices.create({ index: this.indexName });
        this.logger.log(`Elasticsearch index '${this.indexName}' created.`);
      } else {
        this.logger.log(`Elasticsearch index '${this.indexName}' already exists.`);
      }

    } catch (error: any) {
      this.logger.error('Failed to connect or initialize Elasticsearch:', error);
      throw new InternalServerErrorException('Failed to connect to Elasticsearch.');
    }
  }

  async onModuleDestroy() {
    if (this.client) {
      await this.client.close();
      this.logger.log('Elasticsearch connection closed.');
    }
  }

  async indexDocument(id: string, document: Task): Promise<void> {
    try {
      const documentToIndex: Partial<Task> = { ...document };
      delete documentToIndex._id;
      delete documentToIndex._rev;

      await this.client.index({
        index: this.indexName,
        id: id,
        document: documentToIndex,
      });
      this.logger.debug(`Document ${id} indexed in Elasticsearch.`);
    } catch (error: any) {
      this.logger.error(`Failed to index document ${id}:`, error);
      throw new InternalServerErrorException(`Failed to index document ${id}.`);
    }
  }

  async searchDocuments(query: string, userId: string): Promise<Task[]> { 
    try {
      const { hits } = await this.client.search({
        index: this.indexName,
        query: {
          bool: {
            must: [
              {
                multi_match: {
                  query: query,
                  fields: ['title', 'description'], 
                  type: 'phrase_prefix',
                },
              },
              {
                term: {
                  userId: userId, 
                },
              },
            ],
          },
        },
      });
      return hits.hits.map(hit => hit._source as Task);
    } catch (error: any) {
      this.logger.error(`Failed to search documents for query "${query}" and user "${userId}":`, error);
      throw new InternalServerErrorException(`Failed to search documents.`);
    }
  }

  async deleteDocument(id: string): Promise<void> {
    try {
      await this.client.delete({
        index: this.indexName,
        id: id,
      });
      this.logger.debug(`Document ${id} deleted from Elasticsearch.`);
    } catch (error: any) {
      this.logger.error(`Failed to delete document ${id} from Elasticsearch:`, error);
      throw new InternalServerErrorException(`Failed to delete document ${id} from search index.`);
    }
  }
}
