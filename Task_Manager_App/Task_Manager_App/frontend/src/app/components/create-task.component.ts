import { Component, OnInit } from '@angular/core'; 
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';

import { FormGroup, FormBuilder, Validators, ReactiveFormsModule } from '@angular/forms';

export interface Task {
  title: string;
  description?: string;
  status: 'Pending' | 'In Progress' | 'Completed';
}

@Component({
  selector: 'app-create-task',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule], 
  templateUrl: './create-task.component.html',
  styleUrl: './create-task.component.css'
})
export class CreateTaskComponent implements OnInit { 
  taskForm!: FormGroup; 
  loading = false; 
  successMessage: string | null = null; 
  errorMessage: string | null = null; 

  private apiUrl = 'http://localhost:3000/api/tasks';

  constructor(
    private fb: FormBuilder, 
    public router: Router,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    this.taskForm = this.fb.group({
      title: ['', Validators.required], 
      description: [''],
      status: ['Pending', Validators.required] 
    });
  }

  
  async onSubmit(): Promise<void> {
    this.loading = true; 
    this.successMessage = null; 
    this.errorMessage = null;

    if (this.taskForm.invalid) {
      
      this.taskForm.markAllAsTouched();
      this.loading = false;
      this.errorMessage = 'Please fill in all required fields.';
      return;
    }

    try {
      await new Promise<void>((resolve, reject) => {
        this.http.post(this.apiUrl, this.taskForm.value).subscribe({
          next: () => {
            this.successMessage = 'Task created successfully!';
            this.loading = false;
            this.taskForm.reset({
              title: '',
              description: '',
              status: 'Pending'
            });
            setTimeout(() => {
              this.router.navigate(['/view-task']);
              resolve();
            }, 2000);
          },
          error: (err) => {
            console.error('Error creating task:', err);
            this.errorMessage = `Failed to create task: ${err.message || 'Unknown error'}`;
            this.loading = false;
            reject(err);
          }
        });
      });
      
    } catch (err:any) {
      
      console.error('Unhandled error during task creation:', err);
      if (!this.errorMessage) { 
         this.errorMessage = `An unexpected error occurred: ${err.message || 'Unknown error'}`;
      }
      this.loading = false;
    }
  }
}