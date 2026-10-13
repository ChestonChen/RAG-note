# Redis Notes

## Redis Persistence

Redis is an in-memory data store. RDB persistence writes periodic snapshots,
while AOF persistence appends write operations so state can be reconstructed.

## Redis Failover

Redis replication and failover promote a replica when a primary becomes
unavailable, subject to the configured cluster or sentinel behavior.
