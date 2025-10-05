import { Injectable } from '@angular/core';
import { HttpRequest, HttpHandler, HttpEvent, HttpInterceptor, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Router } from '@angular/router';

@Injectable()
export class JwtInterceptor implements HttpInterceptor {

  constructor(private router: Router) {}

  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = localStorage.getItem('access_token');

    if (token) {
      request = request.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`
        }
      });
    }

    return next.handle(request).pipe(
      catchError((error: HttpErrorResponse) => {
        if (error.status === 401) {
          const isLoginOrRegister =
            request.url.includes('/api/auth/login') ||
            request.url.includes('/api/users/register');

          if (!isLoginOrRegister) {
            console.warn('401 Unauthorized response received, logging out...');
            localStorage.removeItem('access_token');
            localStorage.removeItem('currentUser');
            this.router.navigate(['/login']);
          }
        }
        return throwError(() => error);
      })
    );
  }
}