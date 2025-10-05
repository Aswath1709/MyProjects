import { Controller, Post, Body, HttpCode, HttpStatus, ConflictException } from '@nestjs/common';
import { UsersService } from './users.service';
import { CreateUserDto } from './dto/create-user.dto';
import { User } from './interfaces/user.interface'; 

@Controller('users') 
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Post('register') 
  @HttpCode(HttpStatus.CREATED)
  async register(@Body() createUserDto: CreateUserDto): Promise<Omit<User, 'password'>> {
    try {
      const user = await this.usersService.create(createUserDto);

      return user; 
    } catch (error) {
      if (error instanceof ConflictException) {
        throw error; 
      }
      throw new ConflictException('User registration failed.'); 
    }
  }
}
