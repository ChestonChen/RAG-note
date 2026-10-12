# Network Notes

These notes cover transport protocols and a few unrelated storage terms.

## TCP Reliability

TCP establishes a connection, numbers bytes, acknowledges delivery, and
retransmits lost segments. This provides ordered and reliable transport for
applications that need a complete stream.

## UDP Datagram Transport

UDP sends independent datagrams without a connection handshake or built-in
delivery guarantee. It has less overhead and is useful when an application can
tolerate loss or needs low latency.

## Choosing A Protocol

Use TCP for reliable ordered data and UDP for lightweight datagrams. Neither
protocol is a PostgreSQL table or database feature.
