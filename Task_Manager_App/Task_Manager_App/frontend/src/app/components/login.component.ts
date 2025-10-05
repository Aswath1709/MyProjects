import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; 
import { RouterModule } from '@angular/router';
import { AuthService, LoginUserDto } from '../auth/auth.service'; 

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule], 
  templateUrl: './login.component.html',
  styleUrl: './login.component.css'
})
export class LoginComponent {
  loginDto: LoginUserDto = { username: '', password: '' }; 
  errorMessage: string | null = null;
  isLoading = false;

  constructor(private authService: AuthService) {}

  async onLoginSubmit(): Promise<void> {
    this.isLoading = true;
    this.errorMessage = null; 

    try {
      await this.authService.login(this.loginDto);
      
    } catch (error: any) {
      this.errorMessage = error.message || 'An unexpected error occurred during login.';
      console.error('Login error:', error);
    } finally {
      this.isLoading = false;
    }
  }
}
