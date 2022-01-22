package xyz.laur.duels;

import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.*;

public class GamesCommandTest {
    private GamesCommand command;
    private SessionManager sessionManager;

    @Before
    public void setUp() {
        sessionManager = new SessionManager();
        command = new GamesCommand(sessionManager);
    }

    @Test
    public void testOnCommandNoSessions() {
        CommandSender sender = mock(CommandSender.class);
        assertTrue(command.onCommand(sender, null, "games", new String[0]));

        verify(sender, times(1))
            .sendMessage("§eNo active games");
    }

    @Test
    public void testOnCommandTwoSessions() {
        CommandSender sender = mock(CommandSender.class);

        GameSession session1 = mock(GameSession.class);
        when(session1.getMapName()).thenReturn("Space Mine");
        when(session1.getState()).thenReturn(GameState.PLAYING);

        Player player1 = mock(Player.class);
        when(player1.getName()).thenReturn("Player123");

        Player player2 = mock(Player.class);
        when(player2.getName()).thenReturn("Laur");

        when(session1.getPlayers()).thenReturn(Arrays.asList(player1, player2));

        sessionManager.addSession("sess1", session1);

        GameSession session2 = mock(GameSession.class);
        when(session2.getMapName()).thenReturn("White Crystal");
        when(session2.getState()).thenReturn(GameState.WAITING_FOR_PLAYERS);

        Player player3 = mock(Player.class);
        when(player3.getName()).thenReturn("ol0fmeister");

        when(session2.getPlayers()).thenReturn(Collections.singletonList(player3));

        sessionManager.addSession("sess2", session2);

        assertTrue(command.onCommand(sender, null, "games", new String[0]));

        verify(sender, times(1))
            .sendMessage("§esess1: [Player123, Laur], Playing, Space Mine");

        verify(sender, times(1))
            .sendMessage("§6sess2: [ol0fmeister], Waiting for Players, White Crystal");

    }
}
