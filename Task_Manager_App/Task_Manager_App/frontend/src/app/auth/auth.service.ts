import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { BehaviorSubject, Observable } from 'rxjs'; 
import { map } from 'rxjs/operators'; 

export interface LoginUserDto {
  username: string;
  password: string;
}

export interface RegisterUserDto {
  username: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
}

export interface UserProfile {
  userId: string;
  username: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private backendUrl = 'http://localhost:3000/api/auth';
  private userRegistrationUrl = 'http://localhost:3000/api/users/register';

 
  private _currentUserProfileSubject: BehaviorSubject<UserProfile | null>;
 
  public currentUserProfile$: Observable<UserProfile | null>;
  
  public isAuthenticated$: Observable<boolean>;

  constructor(private http: HttpClient, private router: Router) {
    const storedUser = localStorage.getItem('currentUser');
    const initialProfile: UserProfile | null = storedUser ? JSON.parse(storedUser) : null;
    this._currentUserProfileSubject = new BehaviorSubject<UserProfile | null>(initialProfile);
    this.currentUserProfile$ = this._currentUserProfileSubject.asObservable();

    this.isAuthenticated$ = this.currentUserProfile$.pipe(
      map(user => user !== null)
    );

    console.log('AuthService initialized. Initial current user profile:', initialProfile);

    if (this.getToken()) {
      this.checkAuthStatus().catch(error => {
        console.warn('AuthService: Initial token validation failed during startup. User will be logged out if not already.', error);
      });
    } else {
      this._currentUserProfileSubject.next(null);
    }
  }

  public get currentUserValue(): UserProfile | null {
    return this._currentUserProfileSubject.value;
  }

  private setToken(token: string): void {
    localStorage.setItem('access_token', token);
    console.log('Token stored in localStorage.');
  }

  public getToken(): string | null {
    const token = localStorage.getItem('access_token');
    console.log('getToken called. Token present:', !!token);
    return token;
  }

  private removeToken(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('currentUser');
    this._currentUserProfileSubject.next(null); 
    console.log('Token and user profile removed from localStorage. User logged out.');
  }

  private async observableToPromise<T>(observable: Observable<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      observable.subscribe({
        next: (response: T) => resolve(response),
        error: (err: any) => reject(err)
      });
    });
  }

  async register(userDto: RegisterUserDto): Promise<any> {
    try {
      console.log('Attempting to register user:', userDto.username);
      const response = await this.observableToPromise<any>(this.http.post<any>(this.userRegistrationUrl, userDto));
      console.log('Registration successful:', response);

      console.log('Registration successful. Redirecting to login page.');
      this.router.navigate(['/login']);

      return response;
    } catch (error: any) {
      console.error('Registration failed:', error);
      throw new Error(error.error?.message || 'Registration failed');
    }
  }

  async login(userDto: LoginUserDto): Promise<AuthResponse> {
    try {
      console.log('Attempting to log in user:', userDto.username);
      const response = await this.observableToPromise<AuthResponse>(this.http.post<AuthResponse>(`${this.backendUrl}/login`, userDto));
      console.log('Login request successful. Response:', response);
      this.setToken(response.access_token);

      console.log('Attempting to fetch user profile after login.');
      const profile = await this.observableToPromise<UserProfile>(this.http.get<UserProfile>(`${this.backendUrl}/profile`));
      console.log('User profile fetched successfully:', profile);

      localStorage.setItem('currentUser', JSON.stringify(profile));
      this._currentUserProfileSubject.next(profile); 
      console.log('CurrentUserProfile updated. Navigating to /home.');
      this.router.navigate(['/home']);
      return response;
    } catch (error: any) {
      console.error('Login failed:', error);
      this.removeToken();
      throw new Error(error.error?.message || 'Login failed');
    }
  }

  async getProfile(): Promise<UserProfile> {
    try {
      console.log('Sending request to get user profile...');
      const profile = await this.observableToPromise<UserProfile>(this.http.get<UserProfile>(`${this.backendUrl}/profile`));
      console.log('Received user profile:', profile);
      return profile;
    } catch (error: any) {
      console.error('Failed to get user profile:', error);
      throw new Error(error.error?.message || 'Failed to get user profile');
    }
  }

  logout(): void {
    this.removeToken();
    console.log('Redirecting to /login after logout.');
    this.router.navigate(['/login']);
  }

  async checkAuthStatus(): Promise<UserProfile> {
    console.log('AuthService: checkAuthStatus called.');
    const token = this.getToken();
    if (!token) {
      console.warn('AuthService: No token found during checkAuthStatus. Logging out.');
      this.logout();
      throw new Error('No token found');
    }

    try {
      console.log('AuthService: Token found. Attempting to get profile for status check.');
      const profile = await this.getProfile();
      localStorage.setItem('currentUser', JSON.stringify(profile));
      this._currentUserProfileSubject.next(profile); 
      console.log('AuthService: Auth status check successful. User is authenticated.');
      return profile;
    } catch (err: any) {
      console.error('AuthService: Auth status check failed:', err);
      this.logout();
      throw new Error('Authentication check failed');
    }
  }
}
