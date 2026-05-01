class FakeObject:
    def __init__(self, Key, Body, ContentType):
        self._key = Key
        self._body = Body
        self._content_type = ContentType
    
    @property
    def key(self):
        return self._key


class FakeBucketResource:
    def __init__(self, bucket_name):
        self._bucket_name = bucket_name
        self._objects = {}

    def put_object(self, Key, Body, ContentType):
        object_ = FakeObject(Key=Key, Body=Body, ContentType=ContentType)
        self._objects[Key] = object_
        return object_


class FakeS3Resource:
    def Bucket(self, bucket_name):
        return FakeBucketResource(bucket_name)


def fake_boto_resource(resource_type):
    if resource_type == 's3':
        return FakeS3Resource()
