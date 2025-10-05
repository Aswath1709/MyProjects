import { Module, Global } from '@nestjs/common';
import Nano from 'nano';

export const NANO_CLIENT = 'NANO_CLIENT';

@Global()
@Module({
  providers: [
    {
      provide: NANO_CLIENT,
      useFactory: () => {
        const couchUrl = process.env.COUCHDB_URL || 'http://admin:password@localhost:5984';
        console.log('[NanoModule] Using CouchDB URL:', couchUrl);
        return Nano(couchUrl);
      }
    },
  ],
  exports: [NANO_CLIENT],
})
export class NanoModule {}
