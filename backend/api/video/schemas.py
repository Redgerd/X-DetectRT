from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Literal
import uuid # For generating example IDs in comments

# Request Schemas
# user_id is mandatory for a valid upload.
class VideoUploadRequest(BaseModel):
    """
    Schema for the metadata expected during video upload.
    This would typically be sent as part of the multipart/form-data.
    """
    user_id: str = Field(
        ...,
        min_length=1,
        description="The ID of the user uploading the video. This field is required."
    )

# Response Schemas

class VideoUploadResponse(BaseModel):
    """
    Schema for the response after a video is successfully uploaded.
    Corresponds to 201 Created or 202 Accepted.
    """
    video_id: str = Field(
        ...,
        description="Unique identifier for the uploaded video."
    )
    user_id: str = Field( 
        ...,
        description="The ID of the user who owns this video."
    )
    status: Literal["UPLOADED", "PROCESSING", "COMPLETED", "FAILED", "PENDING"] = Field(
        ...,
        description="Current processing status of the video."
    )
    message: str = Field(
        ...,
        description="A human-readable message about the upload/processing status."
    )
    status_check_url: HttpUrl = Field(
        ...,
        description="URL to check the processing status of the video."
    )


class VideoStatusResponse(BaseModel):
    """
    Schema for the response when checking the processing status of a video.
    Corresponds to 200 OK.
    """
    video_id: str = Field(
        ...,
        description="Unique identifier for the video being checked."
    )
    user_id: str = Field( # Now including user_id in the response
        ...,
        description="The ID of the user who owns this video."
    )
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        ...,
        description="Current processing status of the video."
    )
    progress: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Estimated processing progress as a percentage (0-100). Only available when status is 'PROCESSING'."
    )
    message: Optional[str] = Field(
        None,
        description="An optional human-readable message about the current status or any errors."
    )
    error_details: Optional[str] = Field(
        None,
        description="Detailed error message if the status is 'FAILED'."
    )