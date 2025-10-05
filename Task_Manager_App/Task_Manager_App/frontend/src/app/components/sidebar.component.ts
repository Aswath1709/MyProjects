import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router'; 
import { AuthService } from '../auth/auth.service'; 

@Component({
  selector: 'app-sidebar',
  standalone: true, 
  imports: [CommonModule, RouterModule], 
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.css'
})
export class SidebarComponent {
  constructor(private authService: AuthService) {} 

  async onLogout(): Promise<void> {
    console.log('Logout button clicked. Initiating logout...');
    this.authService.logout(); 
  }
}
