export interface Task {
  _id?: string;
  _rev?: string;
  title: string;
  description: string;
  status: 'Pending' | 'In Progress' | 'Completed'; 
  createdAt?: string;
  userId: string;
}