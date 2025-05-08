# YouTube API Integration

## Overview
The YouTube API integration is implemented in the `YouTubeSource` class, which extends the `BaseSource` abstract class. This integration handles fetching and formatting video content from YouTube.

## Implementation Details

### Class Structure
```python
class YouTubeSource(BaseSource):
    """Source for fetching YouTube videos"""
```

### Main Methods

#### 1. Fetch Content
```python
async def fetch_content(self, query: str, limit: int = 5, page_token: str = None) -> Dict[str, Any]:
    """
    Fetch content from YouTube based on the query
    
    Args:
        query: Search query or keywords
        limit: Maximum number of items to return
        page_token: Token for pagination
        
    Returns:
        Dictionary containing videos and next page token
    """
```

### Video Processing

#### 1. Video Details Format
```python
video = {
    "id": item["id"],
    "title": item["snippet"]["title"],
    "description": item["snippet"]["description"],
    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
    "channel": item["snippet"]["channelTitle"],
    "published_at": item["snippet"]["publishedAt"],
    "url": f"https://www.youtube.com/watch?v={item['id']}",
    "embed_url": f"https://www.youtube.com/embed/{item['id']}",
    "duration": duration,
    "view_count": view_count,
    "source": "youtube"
}
```

### Utility Methods

#### 1. Duration Formatting
```python
def _format_duration(self, duration_iso):
    """
    Convert ISO 8601 duration to a readable format
    Example: PT1H30M15S -> 1h 30m 15s
    """
```

#### 2. View Count Formatting
```python
def _format_view_count(self, view_count):
    """
    Format view count to a readable format
    Example: 1234567 -> 1,234,567
    """
```

## Error Handling

### Common Issues
1. **403 Forbidden Error**
   - Invalid API key
   - Expired API key
   - Exceeded API quota
   - Insufficient permissions

2. **Rate Limiting**
   - Implemented caching with `@lru_cache`
   - Cache duration: 1 hour
   - Maximum cache size: 32 entries

## Best Practices

### 1. API Key Management
- Store API key in environment variables
- Rotate keys periodically
- Monitor quota usage

### 2. Error Handling
- Implement fallback to mock data
- Log errors with full traceback
- Cache successful responses

### 3. Performance Optimization
- Use caching for frequent queries
- Batch video detail requests
- Format data efficiently

## Integration Points

### 1. Feed Router
- Used in `/api/feed` endpoint
- Handles pagination
- Implements fallback mechanism

### 2. Source Manager
- Manages multiple content sources
- Provides unified interface
- Handles source selection

## Monitoring and Logging

### 1. Key Metrics
- API response times
- Error rates
- Cache hit rates
- Quota usage

### 2. Logging
- Request details
- Error messages
- Performance metrics
- Cache statistics 
