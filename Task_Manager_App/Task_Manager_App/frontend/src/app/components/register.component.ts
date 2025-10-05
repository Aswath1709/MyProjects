import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { AuthService, RegisterUserDto } from '../auth/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './register.component.html',
  styleUrl: './register.component.css'
})
export class RegisterComponent {
  registerDto: RegisterUserDto = { username: '', password: '' };
  errorMessage: string | null = null;
  successMessage: string | null = null;
  isLoading = false;

  constructor(private authService: AuthService) {}

  async onRegisterSubmit(): Promise<void> {
    this.isLoading = true;
    this.errorMessage = null;
    this.successMessage = null;

    try {
      await this.authService.register(this.registerDto);
      this.successMessage = 'Registration successful! Please log in with your new credentials.';
      this.registerDto = { username: '', password: '' };
    } catch (error: any) {
      this.errorMessage = error.message || 'An unexpected error occurred during registration.';
      console.error('Registration error:', error);
      this.successMessage = null;
    } finally {
      this.isLoading = false;
    }
  }
}
