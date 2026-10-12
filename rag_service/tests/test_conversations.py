from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from rag_service.app.db import get_engine, init_db
from rag_service.app.models import Conversation, Message
from rag_service.app.api import serialize_conversation


def test_conversation_messages_round_trip_in_order_and_cascade():
    engine = init_db(get_engine(database_url="sqlite+pysqlite:///:memory:"))

    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_time = first_time + timedelta(seconds=1)
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        conversation = Conversation(title="Persistent chat")
        session.add(conversation)
        session.flush()
        session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content="second",
                    status="complete",
                    citations=[{"citation_id": "c2", "score": 0.8}],
                    retrieval_debug={"hits": [2]},
                    confidence="medium",
                    confidence_score=0.8,
                    mode="kb_only",
                    web_sources=[{"title": "Source", "url": "https://example.test"}],
                    created_at=second_time,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content="first",
                    status="complete",
                    citations=[{"citation_id": "c1", "score": 0.9}],
                    retrieval_debug={"hits": [1]},
                    confidence=None,
                    confidence_score=None,
                    mode="kb_only",
                    created_at=first_time,
                ),
            ]
        )
        session.commit()

        loaded = session.scalar(select(Conversation).where(Conversation.id == conversation.id))
        assert loaded is not None
        response = serialize_conversation(loaded)
        assert [message.content for message in response.messages] == ["first", "second"]
        assert response.messages[0].citations == [{"citation_id": "c1", "score": 0.9}]
        assert response.messages[1].web_sources == [{"title": "Source", "url": "https://example.test"}]

        session.delete(loaded)
        session.commit()
        assert session.scalars(select(Message)).all() == []
