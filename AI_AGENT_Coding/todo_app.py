import json
import os

# Define the filename for the JSON file
JSON_FILE = 'tasks.json'

def load_tasks():
    """
    Load tasks from the JSON file.
    """
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as file:
            return json.load(file)
    else:
        return []

def save_tasks(tasks):
    """
    Save tasks to the JSON file.
    """
    with open(JSON_FILE, 'w') as file:
        json.dump(tasks, file, indent=4)

def add_task(tasks):
    """
    Add a new task to the list.
    """
    task_name = input("Enter task name: ")
    tasks.append({'name': task_name, 'completed': False})
    save_tasks(tasks)
    print(f"Task '{task_name}' added successfully.")

def delete_task(tasks):
    """
    Delete a task from the list.
    """
    task_name = input("Enter task name to delete: ")
    for task in tasks:
        if task['name'] == task_name:
            tasks.remove(task)
            save_tasks(tasks)
            print(f"Task '{task_name}' deleted successfully.")
            return
    print(f"Task '{task_name}' not found.")

def mark_task_complete(tasks):
    """
    Mark a task as complete.
    """
    task_name = input("Enter task name to mark as complete: ")
    for task in tasks:
        if task['name'] == task_name:
            task['completed'] = True
            save_tasks(tasks)
            print(f"Task '{task_name}' marked as complete.")
            return
    print(f"Task '{task_name}' not found.")

def list_tasks(tasks):
    """
    List all tasks.
    """
    if not tasks:
        print("No tasks available.")
    else:
        print("Tasks:")
        for i, task in enumerate(tasks, start=1):
            status = "Completed" if task['completed'] else "Not Completed"
            print(f"{i}. {task['name']} - {status}")

def main():
    tasks = load_tasks()
    while True:
        print("\nTodo List Manager")
        print("1. Add Task")
        print("2. Delete Task")
        print("3. Mark Task as Complete")
        print("4. List Tasks")
        print("5. Quit")
        choice = input("Enter your choice: ")
        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            delete_task(tasks)
        elif choice == '3':
            mark_task_complete(tasks)
        elif choice == '4':
            list_tasks(tasks)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()