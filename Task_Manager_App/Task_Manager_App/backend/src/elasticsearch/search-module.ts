import { Module, Global } from '@nestjs/common';
import { ElasticsearchService } from './search-service';

@Global() 
@Module({
  providers: [ElasticsearchService],
  exports: [ElasticsearchService], 
})
export class ElasticsearchModule {}