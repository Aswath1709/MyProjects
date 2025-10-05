import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { HttpClient } from '@angular/common/http';

export interface Task {
  _id?: string;
  _rev?: string;
  title: string;
  description: string;
  status: 'Pending' | 'In Progress' | 'Completed';
  createdAt?: string;
}

@Component({
  selector: 'app-update-task',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './update-task.component.html',
  styleUrl: './update-task.component.css'
})
export class UpdateTaskComponent implements OnInit {
  taskId: string | null = null;
  task: Task | null = null;
  isLoading = true;
  error: string | null = null;
  isSaving = false;

  private apiUrl = 'http://localhost:3000/api/tasks';

  constructor(
    private route: ActivatedRoute,
    public router: Router,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.taskId = params.get('id');
      if (this.taskId) {
        this.fetchTask(this.taskId);
      } else {
        this.error = 'No task ID provided for updating.';
        this.isLoading = false;
      }
    });
  }

  async fetchTask(id: string): Promise<void> {
    this.isLoading = true;
    this.error = null;
    try {
      
      const data: Task = await new Promise((resolve, reject) => {
        this.http.get<Task>(`${this.apiUrl}/${id}`).subscribe({
          next: (res) => resolve(res),
          error: (err) => reject(err)
        });
      });
      this.task = data;
      if (this.task && !this.task.status) {
        this.task.status = 'Pending';
      }
      this.isLoading = false;
    } catch (err: any) {
      console.error('Error fetching task for update:', err);
      this.error = `Failed to load task for updating: ${err.message || err.error?.message || 'Unknown error'}`;
      this.isLoading = false;
    }
  }

   async saveTask(): Promise<void> {
    if (!this.task || !this.taskId) {
      this.error = 'Task data is missing.';
      return;
    }

    this.isSaving = true;
    this.error = null;

    try {
      
      const taskData = this.task; 
  

      await new Promise<void>((resolve, reject) => {
        this.http.patch(`${this.apiUrl}/${this.taskId}`, {
          title: taskData.title,       
          description: taskData.description, 
          status: taskData.status     
        }).subscribe({
          next: () => resolve(),
          error: (err) => reject(err)
        });
      });

      this.isSaving = false;
      alert('Task updated successfully!');
      this.router.navigate(['/view-task']);
    } catch (err: any) {
      console.error('Error saving task:', err);
      this.error = `Failed to update task: ${err.message || err.error?.message || 'Unknown error'}`;
      this.isSaving = false;
    }
  }
}