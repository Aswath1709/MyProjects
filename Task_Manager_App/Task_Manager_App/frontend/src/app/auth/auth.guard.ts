import { CanActivateFn, Router } from '@angular/router';
import { inject } from '@angular/core';
import { AuthService } from './auth.service';

export const authGuard: CanActivateFn = async (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  console.log('AuthGuard: Activating for route:', state.url);

  try {
    console.log('AuthGuard: Calling authService.checkAuthStatus()...');
    await authService.checkAuthStatus();
    console.log('AuthGuard: authService.checkAuthStatus() SUCCEEDED. User is authenticated. Allowing access to:', state.url);
    return true;

  } catch (error) {
    console.error('AuthGuard: authService.checkAuthStatus() FAILED.', error);
    console.warn('AuthGuard: User not authenticated or token invalid. Redirecting to login.', error);
 
    router.navigate(['/login'], { queryParams: { returnUrl: state.url } });
    return false; 
  }
};
