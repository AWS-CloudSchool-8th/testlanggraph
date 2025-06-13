import boto3

def load_caption_from_s3(bucket_name: str, key: str) -> str:
    """
    S3에서 caption 파일을 불러와 문자열로 반환
    """
    s3 = boto3.client("s3")

    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read().decode("utf-8")
        return content
    except Exception as e:
        print(f"S3에서 caption을 불러오지 못했습니다: {e}")
        return ""

