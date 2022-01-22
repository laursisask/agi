package xyz.laur.duels;

import org.bukkit.Server;
import org.bukkit.entity.Player;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.mockito.Mockito.*;

public class InvisibilityManagerTest {
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;
    private Server server;

    @Before
    public void setUp() {
        sessionManager = new SessionManager();
        server = mock(Server.class);

        invisibilityManager = new InvisibilityManager(sessionManager, server);
    }

    @Test
    public void testInvisibility() {
        SumoGameSession session1 = mock(SumoGameSession.class);
        SumoGameSession session2 = mock(SumoGameSession.class);
        SumoGameSession session3 = mock(SumoGameSession.class);

        sessionManager.addSession("a", session1);
        sessionManager.addSession("b", session2);
        sessionManager.addSession("c", session3);

        Player player1 = mock(Player.class);
        when(player1.canSee(any(Player.class))).thenReturn(true);

        Player player2 = mock(Player.class);
        when(player2.canSee(any(Player.class))).thenReturn(true);

        Player player3 = mock(Player.class);
        when(player3.canSee(any(Player.class))).thenReturn(true);

        Player player4 = mock(Player.class);
        when(player4.canSee(any(Player.class))).thenReturn(true);

        Player player5 = mock(Player.class);
        when(player5.canSee(any(Player.class))).thenReturn(true);

        Player player6 = mock(Player.class);
        when(player6.canSee(any(Player.class))).thenReturn(true);

        doReturn(Arrays.asList(player1, player2, player3, player4, player5, player6)).when(server).getOnlinePlayers();

        session1.addPlayer(player1);
        session1.addPlayer(player2);
        when(session1.hasPlayer(player1)).thenReturn(true);
        when(session1.hasPlayer(player2)).thenReturn(true);

        when(session2.hasPlayer(player3)).thenReturn(true);
        when(session2.hasPlayer(player4)).thenReturn(true);

        when(session3.hasPlayer(player5)).thenReturn(true);

        invisibilityManager.update();

        verify(player1, never()).hidePlayer(player2);
        verify(player1, times(1)).hidePlayer(player3);
        verify(player1, times(1)).hidePlayer(player4);
        verify(player1, times(1)).hidePlayer(player5);
        verify(player1, times(1)).hidePlayer(player6);

        verify(player2, never()).hidePlayer(player1);
        verify(player2, times(1)).hidePlayer(player3);
        verify(player2, times(1)).hidePlayer(player4);
        verify(player2, times(1)).hidePlayer(player5);
        verify(player2, times(1)).hidePlayer(player6);

        verify(player3, never()).hidePlayer(player4);
        verify(player3, times(1)).hidePlayer(player1);
        verify(player3, times(1)).hidePlayer(player2);
        verify(player3, times(1)).hidePlayer(player5);
        verify(player3, times(1)).hidePlayer(player6);

        verify(player4, never()).hidePlayer(player3);
        verify(player4, times(1)).hidePlayer(player1);
        verify(player4, times(1)).hidePlayer(player2);
        verify(player4, times(1)).hidePlayer(player5);
        verify(player4, times(1)).hidePlayer(player6);

        verify(player5, times(1)).hidePlayer(player1);
        verify(player5, times(1)).hidePlayer(player2);
        verify(player5, times(1)).hidePlayer(player3);
        verify(player5, times(1)).hidePlayer(player4);
        verify(player5, times(1)).hidePlayer(player6);

        verify(player6, never()).hidePlayer(any(Player.class));
    }
}
