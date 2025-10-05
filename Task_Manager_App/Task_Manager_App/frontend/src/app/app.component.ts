import { Component, OnInit } from '@angular/core';
import { Router, RouterModule, Event, NavigationStart, NavigationEnd, NavigationCancel, NavigationError } from '@angular/router';
import { CommonModule } from '@angular/common';
import { SidebarComponent } from './components/sidebar.component';
import { AuthService } from './auth/auth.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
  standalone: true,
  imports: [
    CommonModule,
    SidebarComponent,
    RouterModule
  ],
})
export class AppComponent implements OnInit {
  title = 'frontend';

  isAuthenticated$: Observable<boolean>;

  constructor(private router: Router, private authService: AuthService) {
    this.isAuthenticated$ = this.authService.isAuthenticated$;

    this.router.events.subscribe((event: Event) => {
      if (event instanceof NavigationStart) {
        console.log('Router Event: NavigationStart', event.url);
      }
      if (event instanceof NavigationEnd) {
        console.log('Router Event: NavigationEnd', event.url);
      }
      if (event instanceof NavigationCancel) {
        console.warn('Router Event: NavigationCancel', event.url, event.reason);
      }
      if (event instanceof NavigationError) {
        console.error('Router Event: NavigationError', event.url, event.error);
      }
    });
  }

  ngOnInit(): void {
    void 0;
  }
}