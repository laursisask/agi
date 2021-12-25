package xyz.laur.duels;

import org.bukkit.Location;
import org.bukkit.Server;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.PluginManager;
import org.bukkit.scheduler.BukkitScheduler;
import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.*;

public class GameSessionTest {
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;
    private GameSession session;
    private World world;
    private BukkitScheduler scheduler;

    @Before
    public void setUp() {
        sessionManager = new SessionManager();
        invisibilityManager = mock(InvisibilityManager.class);
        Plugin plugin = mock(Plugin.class);
        when(plugin.getLogger()).thenReturn(Logger.getGlobal());

        Server server = mock(Server.class);
        PluginManager pluginManager = mock(PluginManager.class);
        scheduler = mock(BukkitScheduler.class);
        when(server.getPluginManager()).thenReturn(pluginManager);
        when(server.getScheduler()).thenReturn(scheduler);
        when(plugin.getServer()).thenReturn(server);

        world = mock(World.class);

        session = new GameSession(world, sessionManager, invisibilityManager, plugin, 0F, GameSession.GameMap.PONSEN);
    }

    @Test
    public void testGameMainLifecycle() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        verify(invisibilityManager, times(1)).update();

        assertEquals(GameSession.State.WAITING_FOR_PLAYERS, session.getState());
        assertTrue(session.hasPlayer(player1));
        verify(player1, never()).teleport(any(Location.class));

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameSession.State.PLAYING, session.getState());
        assertTrue(session.hasPlayer(player1));
        assertTrue(session.hasPlayer(player2));
        verify(player1, times(1)).sendMessage("Game started");
        verify(player2, times(1)).sendMessage("Game started");

        verify(invisibilityManager, times(2)).update();

        verify(player1, times(1)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));

        verify(player1.getInventory(), times(1)).clear();
        verify(player2.getInventory(), times(1)).clear();

        Player unrelatedPlayer = createMockPlayer();
        Location from = new Location(world, 10, 150, 20);
        Location to = new Location(world, 10, 3, 23);
        session.onMove(new PlayerMoveEvent(unrelatedPlayer, from, to));
        assertEquals(GameSession.State.PLAYING, session.getState());

        session.onMove(new PlayerMoveEvent(player1, from, to));
        assertEquals(GameSession.State.ENDED, session.getState());
        verify(player1, times(1)).sendMessage("You lost");
        verify(player2, times(1)).sendMessage("You won");
        verify(invisibilityManager, times(3)).update();
    }

    @Test
    public void testPlayerQuits() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameSession.State.PLAYING, session.getState());

        session.onQuit(new PlayerQuitEvent(player2, null));
        assertEquals(GameSession.State.ENDED, session.getState());
        verify(player1, times(1)).sendMessage("You won");
        verify(player2, times(1)).sendMessage("You lost");
    }

    @Test
    public void testTimeout() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameSession.State.PLAYING, session.getState());

        ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
        verify(scheduler, times(1))
                .scheduleSyncDelayedTask(any(Plugin.class), task.capture(), eq(GameSession.MAX_DURATION));

        task.getValue().run();
        assertEquals(GameSession.State.ENDED, session.getState());
        verify(player1, times(1)).sendMessage("You lost");
        verify(player2, times(1)).sendMessage("You lost");
    }

    @Test
    public void testMetadata() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getDamager()).thenReturn(player1);
        when(damageEvent.getEntity()).thenReturn(player2);

        session.onPlayerDamage(damageEvent);

        verify(player1, times(1)).sendMessage("metadata:hit:1.00000");
        verify(player2, times(1)).sendMessage("metadata:hit:-1.00000");
    }

    protected Player createMockPlayer() {
        Player player = mock(Player.class);
        PlayerInventory inventory = mock(PlayerInventory.class);
        when(player.getInventory()).thenReturn(inventory);

        return player;
    }
}
