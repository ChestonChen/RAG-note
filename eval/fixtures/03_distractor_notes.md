# PostgreSQL Administration Notes

This file contains database material that can distract queries about vectors
or networks.

## Tables And Schemas

PostgreSQL tables hold rows and columns. A schema groups related tables, and a
database contains schemas, roles, indexes, and other objects.

## Backups

Logical backups can be created with pg_dump and restored into another
PostgreSQL database. A backup is not a vector embedding and does not describe
TCP or UDP transport.

## Routine Maintenance

VACUUM reclaims dead tuples and ANALYZE refreshes planner statistics. These
operations help ordinary relational tables but are unrelated to nearest
neighbor ranking.
