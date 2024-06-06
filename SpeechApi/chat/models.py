from django.db import models
import uuid

class AudioFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.CharField(max_length=255)
    is_fake = models.BooleanField(default=False)

    def __str__(self):
        return self.name
