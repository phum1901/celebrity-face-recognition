from locust import HttpUser, TaskSet, task



class RecognizerLoadTest(HttpUser):
    @task
    def recognize_test(self):
        with open('tests/1853565-00091.jpg', 'rb') as img:
            self.client.post('/recognize', files={"image": img})