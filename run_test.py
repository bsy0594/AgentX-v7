import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, DataPart

async def main():
    async with httpx.AsyncClient(timeout=3600) as client:
        resolver = A2ACardResolver(httpx_client=client, base_url='http://localhost:9009')
        card = await resolver.get_agent_card()
        print(f"Agent     : {card.name}")
        print(f"Version   : {card.version}")
        print(f"Skills    : {[s.name for s in card.skills]}")
        print()

        factory = ClientFactory(ClientConfig(httpx_client=client, streaming=True))
        a2a = factory.create(card)

        payload = '{"participants": {"agent": "http://localhost:8000/"}, "config": {"competition_id": "spaceship-titanic"}}'
        msg = Message(
            kind='message',
            role=Role.user,
            parts=[Part(TextPart(text=payload))],
            message_id=uuid4().hex,
        )

        print("Sending evaluation request (streaming)...")
        print("-" * 60)
        async for event in a2a.send_message(msg):
            match event:
                case (task, TaskStatusUpdateEvent() as update):
                    state = update.status.state
                    text = ""
                    if update.status.message:
                        for p in update.status.message.parts:
                            if hasattr(p.root, 'text'):
                                text = p.root.text
                    print(f"[{state}] {text}")

                case (task, TaskArtifactUpdateEvent() as update):
                    print(f"\n{'='*60}")
                    print(f"RESULT ARTIFACT: {update.artifact.name}")
                    for p in update.artifact.parts:
                        if isinstance(p.root, TextPart):
                            print(p.root.text)
                        elif isinstance(p.root, DataPart):
                            data = p.root.data
                            print(f"Score       : {data.get('score')}")
                            print(f"Gold Median : {data.get('gold_median')}")
                            print(f"Any Medal   : {data.get('any_medal')}")
                            print(f"Bronze+     : {data.get('above_median')}")
                            for k, v in data.items():
                                if k not in ('score', 'gold_median', 'any_medal', 'above_median'):
                                    print(f"{k}: {v}")
                    print('='*60)

                case _:
                    pass

asyncio.run(main())
