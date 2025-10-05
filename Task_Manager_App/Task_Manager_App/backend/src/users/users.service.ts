import { Injectable, ConflictException, Inject, InternalServerErrorException} from '@nestjs/common';
import * as bcrypt from 'bcrypt';
import * as nano from 'nano'; 
import { User } from './interfaces/user.interface'; 
import { CreateUserDto } from './dto/create-user.dto';
import { NANO_CLIENT } from '../nano/nano.module'; 

@Injectable()
export class UsersService {
  private usersDb: nano.DocumentScope<User>; 
  private readonly usersDbName = 'users'; 

  constructor(@Inject(NANO_CLIENT) private readonly nanoClient: nano.ServerScope) {
    
    this.usersDb = this.nanoClient.use(this.usersDbName);
    this.initializeUserDb(); 
  }

  
  private async initializeUserDb(): Promise<void> {
    try {
      await this.nanoClient.db.get(this.usersDbName);
      console.log(`CouchDB database '${this.usersDbName}' already exists or was created.`);
    } catch (error: any) {
      if (error.statusCode === 404) {
        try {
          await this.nanoClient.db.create(this.usersDbName);
          console.log(`CouchDB database '${this.usersDbName}' created successfully.`);
        } catch (createError: any) {
          console.error(`Failed to create CouchDB database '${this.usersDbName}':`, createError);
          throw new InternalServerErrorException(`Failed to initialize user database.`);
        }
      } else {
        console.error(`Error checking CouchDB database '${this.usersDbName}':`, error);
        throw new InternalServerErrorException(`Failed to connect to user database.`);
      }
    }
  }

 
  private async hashPassword(password: string): Promise<string> {
    const saltRounds = 10;
    return bcrypt.hash(password, saltRounds);
  }

 
  async comparePasswords(password: string, hash: string): Promise<boolean> {
    return bcrypt.compare(password, hash);
  }

  
  async create(createUserDto: CreateUserDto): Promise<Omit<User, 'password'>> {
    
    const existingUsers = await this.usersDb.find({
        selector: {
            username: createUserDto.username
        },
        limit: 1
    });

    if (existingUsers.docs.length > 0) {
        throw new ConflictException('Username already exists');
    }

    const hashedPassword = await this.hashPassword(createUserDto.password);
    const userDoc: User = {
      username: createUserDto.username,
      password: hashedPassword, 
    };

    try {
      const response = await this.usersDb.insert(userDoc);
      if (!response.ok) {
        throw new InternalServerErrorException('Failed to create user in CouchDB.');
      }
      
      const { password, ...result } = { ...userDoc, _id: response.id, _rev: response.rev };
      return result;
    } catch (error: any) {
      if (error.statusCode === 409) {
        throw new ConflictException('Username already exists (conflict during insert).');
      }
      console.error('Error creating user in CouchDB:', error);
      throw new InternalServerErrorException('Failed to create user.');
    }
  }

  async findOneByUsername(username: string): Promise<User | undefined> {
    try {
     
      const result = await this.usersDb.find({
        selector: {
          username: username,
        },
        limit: 1, 
      });

      if (result.docs.length > 0) {
        return result.docs[0]; 
      }
      return undefined; 
    } catch (error: any) {
      
      if (error.statusCode === 404) {
       
        console.warn(`User database '${this.usersDbName}' not found or inaccessible during findOneByUsername.`);
        return undefined;
      }
      console.error(`Error finding user by username '${username}' in CouchDB:`, error);
      throw new InternalServerErrorException(`Failed to find user.`);
    }
  }
}