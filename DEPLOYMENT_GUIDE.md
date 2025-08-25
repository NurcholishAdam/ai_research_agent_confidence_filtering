# AI Research Agent Extensions - Deployment Guide

Complete deployment guide for production environments, including Docker, cloud platforms, and scaling strategies.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Deployment](#docker-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Scaling Strategies](#scaling-strategies)
- [Security Considerations](#security-considerations)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB SSD
- Python: 3.8+
- OS: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: Optional (for diffusion repair acceleration)
- Network: High-bandwidth connection for multi-source fusion

### Dependencies

```bash
# Core dependencies
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install networkx>=2.8
pip install jinja2>=3.1.0
pip install pyyaml>=6.0
pip install psutil>=5.9.0

# Optional dependencies for enhanced features
pip install diffusers>=0.20.0  # For diffusion repair
pip install chromadb>=0.4.0    # For vector storage
pip install pymongo>=4.0       # For RLHF data storage
```

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r extensions/requirements.txt
```

### 2. Configuration

```bash
# Create configuration directory
mkdir -p extensions/config

# Copy example configurations
cp extensions/config/integration_config.example.json extensions/config/integration_config.json
cp extensions/config/observability_config.example.json extensions/config/observability_config.json

# Edit configurations as needed
nano extensions/config/integration_config.json
```

### 3. Initialize Extensions

```python
# test_setup.py
import asyncio
from extensions.integration_orchestrator import integrate_ai_research_agent_extensions

async def test_setup():
    extensions = await integrate_ai_research_agent_extensions()
    print(f"Setup successful! Stages initialized: {len(extensions.initialized_stages)}")

if __name__ == "__main__":
    asyncio.run(test_setup())
```

```bash
python test_setup.py
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt extensions/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p extensions/data extensions/logs extensions/config

# Set environment variables
ENV PYTHONPATH=/app
ENV EXTENSIONS_CONFIG_PATH=/app/extensions/config

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from extensions.integration_orchestrator import AIResearchAgentExtensions; print('healthy')"

# Run application
CMD ["python", "-m", "extensions.integration_orchestrator"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-research-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - EXTENSIONS_CONFIG_PATH=/app/extensions/config
    volumes:
      - ./extensions/data:/app/extensions/data
      - ./extensions/logs:/app/extensions/logs
      - ./extensions/config:/app/extensions/config
    depends_on:
      - mongodb
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from extensions.integration_orchestrator import AIResearchAgentExtensions; print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3

  mongodb:
    image: mongo:5.0
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  mongodb_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 3. Build and Deploy

```bash
# Build image
docker build -t ai-research-agent-extensions .

# Run with docker-compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ai-research-agent
```

## Cloud Platform Deployment

### AWS Deployment

#### 1. ECS with Fargate

```yaml
# aws-task-definition.json
{
  "family": "ai-research-agent-extensions",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ai-research-agent",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ai-research-agent-extensions:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHONPATH",
          "value": "/app"
        },
        {
          "name": "EXTENSIONS_CONFIG_PATH",
          "value": "/app/extensions/config"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-research-agent-extensions",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python -c \"from extensions.integration_orchestrator import AIResearchAgentExtensions; print('healthy')\""
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 2. CloudFormation Template

```yaml
# cloudformation-template.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AI Research Agent Extensions Infrastructure'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID for deployment
  
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs for deployment

Resources:
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: ai-research-agent-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: ai-research-agent-alb
      Scheme: internet-facing
      Type: application
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for ALB
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  # RDS for persistent storage
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for RDS
      SubnetIds: !Ref SubnetIds

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: ai-research-agent-db
      DBInstanceClass: db.t3.medium
      Engine: postgres
      EngineVersion: '13.7'
      MasterUsername: postgres
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 100
      StorageType: gp2
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup

  # ElastiCache for caching
  CacheSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: Subnet group for ElastiCache
      SubnetIds: !Ref SubnetIds

  CacheCluster:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheNodeType: cache.t3.micro
      Engine: redis
      NumCacheNodes: 1
      CacheSubnetGroupName: !Ref CacheSubnetGroup
      VpcSecurityGroupIds:
        - !Ref CacheSecurityGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt LoadBalancer.DNSName
```

### Google Cloud Platform (GCP)

#### 1. Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ai-research-agent-extensions', '.']
  
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ai-research-agent-extensions']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'ai-research-agent-extensions'
      - '--image'
      - 'gcr.io/$PROJECT_ID/ai-research-agent-extensions'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '4Gi'
      - '--cpu'
      - '2'
      - '--max-instances'
      - '10'

images:
  - 'gcr.io/$PROJECT_ID/ai-research-agent-extensions'
```

#### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-research-agent-extensions
  labels:
    app: ai-research-agent-extensions
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-research-agent-extensions
  template:
    metadata:
      labels:
        app: ai-research-agent-extensions
    spec:
      containers:
      - name: ai-research-agent
        image: gcr.io/PROJECT_ID/ai-research-agent-extensions:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: EXTENSIONS_CONFIG_PATH
          value: "/app/extensions/config"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: ai-research-agent-service
spec:
  selector:
    app: ai-research-agent-extensions
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### 1. Container Instances

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "ai-research-agent-extensions"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-03-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "ai-research-agent",
            "properties": {
              "image": "your-registry.azurecr.io/ai-research-agent-extensions:latest",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "PYTHONPATH",
                  "value": "/app"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        },
        "restartPolicy": "Always"
      }
    }
  ]
}
```

## Production Configuration

### 1. Environment Variables

```bash
# Production environment variables
export PYTHONPATH=/app
export EXTENSIONS_CONFIG_PATH=/app/extensions/config
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export METRICS_PORT=9090

# Database connections
export MONGODB_URL=mongodb://username:password@host:port/database
export REDIS_URL=redis://host:port/0

# API Keys (use secrets management in production)
export GROQ_API_KEY=your_groq_key
export OPENAI_API_KEY=your_openai_key

# Performance tuning
export MAX_WORKERS=4
export MEMORY_LIMIT=4GB
export TIMEOUT=300
```

### 2. Production Configuration Files

```json
// extensions/config/production_config.json
{
  "enable_observability": true,
  "enable_context_engineering": true,
  "enable_semantic_graph": true,
  "enable_diffusion_repair": true,
  "enable_rlhf": true,
  "enable_synergies": true,
  "integration_level": "advanced",
  "auto_optimization": true,
  "performance_monitoring": true,
  "production_mode": true,
  "logging": {
    "level": "INFO",
    "format": "json",
    "output": "file",
    "rotation": "daily",
    "retention_days": 30
  },
  "security": {
    "enable_auth": true,
    "rate_limiting": true,
    "input_validation": true,
    "output_sanitization": true
  },
  "performance": {
    "max_concurrent_requests": 100,
    "request_timeout": 300,
    "memory_limit": "4GB",
    "cpu_limit": "2000m"
  }
}
```

### 3. Logging Configuration

```yaml
# logging.yml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /app/extensions/logs/application.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: /app/extensions/logs/error.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  extensions:
    level: INFO
    handlers: [console, file, error_file]
    propagate: no

root:
  level: INFO
  handlers: [console, file]
```

## Monitoring and Observability

### 1. Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'ai-research-agent'
    static_configs:
      - targets: ['ai-research-agent:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "AI Research Agent Extensions",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Stage Performance",
        "type": "table",
        "targets": [
          {
            "expr": "avg_over_time(stage_execution_time[5m])",
            "legendFormat": "{{stage}}"
          }
        ]
      }
    ]
  }
}
```

### 3. Health Checks

```python
# health_check.py
import asyncio
from extensions.integration_orchestrator import AIResearchAgentExtensions

async def health_check():
    """Comprehensive health check"""
    try:
        extensions = AIResearchAgentExtensions()
        
        # Check initialization
        status = await extensions.initialize_all_stages()
        if status['success_rate'] < 0.8:
            return False, "Low initialization success rate"
        
        # Check basic functionality
        test_request = {
            "type": "research",
            "query": "health check test",
            "session_id": "health_check"
        }
        
        result = await extensions.process_enhanced_request(test_request)
        if not result.get('success'):
            return False, "Request processing failed"
        
        # Check performance dashboard
        dashboard = extensions.get_performance_dashboard()
        if not dashboard:
            return False, "Performance dashboard unavailable"
        
        return True, "All systems operational"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    healthy, message = asyncio.run(health_check())
    print(f"Health: {'OK' if healthy else 'FAIL'} - {message}")
    exit(0 if healthy else 1)
```

## Scaling Strategies

### 1. Horizontal Scaling

```yaml
# kubernetes-hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-research-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-research-agent-extensions
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 2. Load Balancing

```nginx
# nginx.conf
upstream ai_research_agent {
    least_conn;
    server ai-research-agent-1:8000 max_fails=3 fail_timeout=30s;
    server ai-research-agent-2:8000 max_fails=3 fail_timeout=30s;
    server ai-research-agent-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://ai_research_agent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    location /health {
        access_log off;
        proxy_pass http://ai_research_agent/health;
    }
}
```

### 3. Caching Strategy

```python
# caching.py
import redis
import json
from typing import Any, Optional

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value"""
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, default=str)
            return self.redis_client.setex(key, ttl, serialized)
        except Exception:
            return False
    
    def cache_request(self, request_hash: str, result: Any, ttl: int = 1800):
        """Cache request result"""
        cache_key = f"request:{request_hash}"
        return self.set(cache_key, result, ttl)
    
    def get_cached_request(self, request_hash: str) -> Optional[Any]:
        """Get cached request result"""
        cache_key = f"request:{request_hash}"
        return self.get(cache_key)
```

## Security Considerations

### 1. Authentication and Authorization

```python
# security.py
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_token(self, user_id: str, permissions: List[str]) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def rate_limit_check(self, user_id: str, endpoint: str, limit: int = 100) -> bool:
        """Check rate limiting"""
        # Implementation depends on your rate limiting strategy
        # Could use Redis, database, or in-memory store
        pass
```

### 2. Input Validation

```python
# validation.py
import re
from typing import Any, Dict, List
from pydantic import BaseModel, validator

class RequestValidation(BaseModel):
    query: str
    session_id: str
    type: str
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:
            raise ValueError('Query too long')
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially dangerous content')
        return v
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid session ID format')
        return v
    
    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['research', 'code_repair', 'analysis']
        if v not in allowed_types:
            raise ValueError(f'Invalid request type. Allowed: {allowed_types}')
        return v
```

### 3. Network Security

```yaml
# network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-research-agent-netpol
spec:
  podSelector:
    matchLabels:
      app: ai-research-agent-extensions
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: load-balancer
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: mongodb
    ports:
    - protocol: TCP
      port: 27017
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## Backup and Recovery

### 1. Data Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup MongoDB
mongodump --host mongodb:27017 --out $BACKUP_DIR/$DATE/mongodb

# Backup Redis
redis-cli --rdb $BACKUP_DIR/$DATE/redis_dump.rdb

# Backup application data
tar -czf $BACKUP_DIR/$DATE/extensions_data.tar.gz /app/extensions/data

# Backup configurations
tar -czf $BACKUP_DIR/$DATE/extensions_config.tar.gz /app/extensions/config

# Cleanup old backups
find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

### 2. Disaster Recovery Plan

```yaml
# disaster-recovery.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
data:
  recovery-steps: |
    1. Assess the situation and determine scope of failure
    2. Activate backup infrastructure if needed
    3. Restore data from most recent backup
    4. Verify data integrity
    5. Restart services in correct order
    6. Run health checks
    7. Gradually restore traffic
    8. Monitor for issues
    9. Document incident and lessons learned
  
  rto: "4 hours"  # Recovery Time Objective
  rpo: "1 hour"   # Recovery Point Objective
  
  contacts: |
    Primary: ops-team@company.com
    Secondary: engineering@company.com
    Emergency: +1-555-0123
```

### 3. Automated Recovery

```python
# recovery.py
import asyncio
import subprocess
from datetime import datetime
from typing import List, Tuple

class DisasterRecovery:
    def __init__(self):
        self.recovery_steps = [
            self.check_system_health,
            self.restore_database,
            self.restore_application_data,
            self.restart_services,
            self.verify_functionality,
            self.restore_traffic
        ]
    
    async def execute_recovery(self) -> bool:
        """Execute disaster recovery plan"""
        print(f"Starting disaster recovery at {datetime.now()}")
        
        for i, step in enumerate(self.recovery_steps, 1):
            print(f"Step {i}: {step.__name__}")
            try:
                success = await step()
                if not success:
                    print(f"Recovery failed at step {i}")
                    return False
                print(f"Step {i} completed successfully")
            except Exception as e:
                print(f"Step {i} failed with error: {e}")
                return False
        
        print("Disaster recovery completed successfully")
        return True
    
    async def check_system_health(self) -> bool:
        """Check system health and determine recovery needs"""
        # Implementation depends on your monitoring setup
        return True
    
    async def restore_database(self) -> bool:
        """Restore database from backup"""
        try:
            # Find latest backup
            result = subprocess.run([
                "find", "/backups", "-name", "mongodb", "-type", "d"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return False
            
            # Restore from latest backup
            latest_backup = result.stdout.strip().split('\n')[-1]
            restore_cmd = [
                "mongorestore", "--host", "mongodb:27017", 
                "--drop", latest_backup
            ]
            
            result = subprocess.run(restore_cmd, capture_output=True)
            return result.returncode == 0
            
        except Exception:
            return False
    
    async def restore_application_data(self) -> bool:
        """Restore application data"""
        # Implementation for restoring application-specific data
        return True
    
    async def restart_services(self) -> bool:
        """Restart all services"""
        services = ["ai-research-agent", "mongodb", "redis"]
        
        for service in services:
            try:
                result = subprocess.run([
                    "docker-compose", "restart", service
                ], capture_output=True)
                
                if result.returncode != 0:
                    return False
            except Exception:
                return False
        
        return True
    
    async def verify_functionality(self) -> bool:
        """Verify system functionality"""
        from health_check import health_check
        
        healthy, _ = await health_check()
        return healthy
    
    async def restore_traffic(self) -> bool:
        """Gradually restore traffic"""
        # Implementation for traffic restoration
        return True
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

**Problem**: Out of memory errors
```
MemoryError: Unable to allocate array
```

**Solutions**:
```bash
# Increase container memory limits
docker run --memory=8g ai-research-agent-extensions

# Optimize memory usage
export MEMORY_LIMIT=6GB
export MAX_CONTEXT_TOKENS=4000

# Enable memory monitoring
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

#### 2. Performance Issues

**Problem**: Slow response times

**Solutions**:
```bash
# Enable performance profiling
export ENABLE_PROFILING=true

# Optimize context packing
export CONTEXT_PACKING_STRATEGY=relevance_first
export MAX_MEMORY_ITEMS=10

# Use caching
export ENABLE_CACHING=true
export CACHE_TTL=3600
```

#### 3. Database Connection Issues

**Problem**: Database connection failures

**Solutions**:
```bash
# Check database connectivity
nc -zv mongodb 27017

# Verify credentials
mongo --host mongodb:27017 --username admin --password

# Check connection pool settings
export DB_POOL_SIZE=10
export DB_TIMEOUT=30
```

#### 4. Integration Failures

**Problem**: Stage initialization failures

**Solutions**:
```python
# Debug initialization
import logging
logging.basicConfig(level=logging.DEBUG)

from extensions.integration_orchestrator import AIResearchAgentExtensions

extensions = AIResearchAgentExtensions()
status = await extensions.initialize_all_stages()

# Check specific stage issues
for stage, config in extensions.integration_status.items():
    if config.get('status') == 'failed':
        print(f"Stage {stage} failed: {config.get('error')}")
```

### Monitoring and Alerting

```yaml
# alerts.yml
groups:
- name: ai-research-agent-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes > 4e9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
      description: "Memory usage is {{ $value | humanize }}B"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Service is down
      description: "{{ $labels.instance }} has been down for more than 1 minute"
```

### Performance Tuning

```bash
# performance-tuning.sh

# CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Memory optimization
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=131072

# Python optimization
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Application-specific tuning
export MAX_CONCURRENT_REQUESTS=50
export REQUEST_TIMEOUT=120
export CONTEXT_CACHE_SIZE=1000
export GRAPH_CACHE_SIZE=5000
```

---

This deployment guide provides comprehensive instructions for deploying the AI Research Agent Extensions in various environments, from local development to production cloud platforms. Follow the appropriate sections based on your deployment needs and infrastructure requirements.

For additional support, refer to the [API Reference](API_REFERENCE.md) and [README](README.md) documentation.