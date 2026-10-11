import unittest

from cloudflare_turn import select_servers


class TurnConfigurationTests(unittest.TestCase):
    def test_filters_ports_and_preserves_temporary_credentials(self):
        result = select_servers(
            {
                "iceServers": [
                    {"urls": "stun:stun.cloudflare.com:3478"},
                    {
                        "urls": [
                            "turn:turn.cloudflare.com:3478?transport=udp",
                            "turns:turn.cloudflare.com:443?transport=tcp",
                            "turn:turn.cloudflare.com:53?transport=udp",
                        ],
                        "username": "temporary-user",
                        "credential": "temporary-password",
                    },
                ]
            }
        )
        self.assertEqual(len(result), 2)
        self.assertNotIn("credential", result[0])
        self.assertEqual(len(result[1]["urls"]), 2)
        self.assertEqual(result[1]["credential"], "temporary-password")

    def test_rejects_missing_credentials_or_no_relay(self):
        for payload in (
            {},
            {"iceServers": [{"urls": "turn:turn.cloudflare.com:3478?transport=udp"}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                select_servers(payload)
