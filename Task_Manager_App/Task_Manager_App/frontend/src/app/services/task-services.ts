import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Task } from '../task.interface'; 

@Injectable({
  providedIn: 'root' 
})
export class TaskService {
 
  private apiUrl = 'http://localhost:3000/api/tasks'; 

  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json'
    })
  };

  constructor(private http: HttpClient) { }

 
  createTask(task: Omit<Task, 'id' | 'createdAt'>): Observable<Task> {
    return this.http.post<Task>(this.apiUrl, task, this.httpOptions);
  }

}