import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

export interface Task {
  _id?: string;
  _rev?: string;
  title: string;
  description: string;
  status: 'Pending' | 'In Progress' | 'Completed';
  createdAt?: string;
  userId: string; 
}

@Component({
  selector: 'app-view-task',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './view-task.component.html',
  styleUrl: './view-task.component.css'
})
export class ViewTaskComponent implements OnInit {
  tasks: Task[] = [];
  isLoading = true;
  error: string | null = null;
  searchQuery = '';

  private apiUrl = 'http://localhost:3000/api/tasks';

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadTasks();
  }

  async loadTasks(): Promise<void> {
    this.isLoading = true;
    this.error = null;

    try {
      let data: Task[];
      if (this.searchQuery.trim()) {
        data = await new Promise<Task[]>((resolve, reject) => {
          this.http.get<Task[]>(`${this.apiUrl}/search?q=${encodeURIComponent(this.searchQuery)}`).subscribe({
            next: (res) => resolve(res),
            error: (err) => reject(err),
          });
        });
      } else {
        data = await new Promise<Task[]>((resolve, reject) => {
          this.http.get<Task[]>(this.apiUrl).subscribe({
            next: (res) => resolve(res),
            error: (err) => reject(err),
          });
        });
      }
      this.tasks = data;
      this.isLoading = false;
    } catch (err: any) {
      console.error('Error loading tasks:', err);
      this.error = `Failed to load tasks: ${err.message || err.error?.message || 'Unknown error'}`;
      this.isLoading = false;
    }
  }

  async deleteTask(task: Task): Promise<void> {
    if (!task._id) {
      console.error('Task ID is missing for deletion.', task);
      this.error = 'Cannot delete task: Missing ID.';
      return;
    }

    if (!confirm(`Are you sure you want to delete the task: "${task.title}"?`)) {
      return;
    }

    try {
      await new Promise<void>((resolve, reject) => {
        this.http.delete(`${this.apiUrl}/${task._id}`).subscribe({
          next: () => resolve(),
          error: (err) => reject(err)
        });
      });

      this.tasks = this.tasks.filter(t => t._id !== task._id);
      console.log(`Task ${task._id} deleted successfully.`);
      this.error = null;
    } catch (err: any) {
      console.error('Error deleting task:', err);
      this.error = `Failed to delete task: ${err.message || err.error?.message || 'Unknown error'}`;
    }
  }
}
