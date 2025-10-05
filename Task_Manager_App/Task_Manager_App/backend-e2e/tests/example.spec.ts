import { test, expect } from '@playwright/test';

test('homepage loads', async ({ page }) => {
  await page.goto('http://localhost:4200');
  await expect(page).toHaveTitle(/Welcome to TaskManager Pro/i); 
});