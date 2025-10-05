import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { TasksModule } from '../tasks/tasks.module';
import { ElasticsearchModule } from '../elasticsearch/search-module';
import { UsersModule } from '../users/users.module';
import { AuthModule } from '../auth/auth.module';
import { NanoModule } from '../nano/nano.module';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env.local', '.env'],
    }),
    TasksModule,
    ElasticsearchModule,
    UsersModule,
    AuthModule,
    NanoModule,
  ],
  controllers: [AppController],
  providers: [
    AppService,
  ],
})
export class AppModule {}