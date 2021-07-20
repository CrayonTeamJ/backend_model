from celery import Celery


app = Celery('tasks', 
                    broker='amqp://admin:admin@localhost//')

@app.task
def add(x, y):
        return x + y