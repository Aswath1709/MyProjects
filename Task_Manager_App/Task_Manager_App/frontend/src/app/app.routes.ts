import { Routes } from '@angular/router';
import { HomeComponent } from './components/home.component'; 
import { CreateTaskComponent } from './components/create-task.component'; 
import { ViewTaskComponent } from './components/view-task.component'; 
import { UpdateTaskComponent } from './components/update-task.component'; 
import { LoginComponent } from './components/login.component'; 
import { RegisterComponent } from './components/register.component'; 
import { authGuard } from './auth/auth.guard'; 

export const routes: Routes = [
  
  { path: '', redirectTo: 'login', pathMatch: 'full' },

 
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },

  
  
  { path: 'home', component: HomeComponent, canActivate: [authGuard] }, 
  { path: 'create-task', component: CreateTaskComponent, canActivate: [authGuard] }, 
  { path: 'view-task', component: ViewTaskComponent, canActivate: [authGuard] }, 
  { path: 'update-task/:id', component: UpdateTaskComponent, canActivate: [authGuard] }, 

  
  { path: '**', redirectTo: 'login' }
];
