package xyz.laur.duels;

import org.bukkit.Server;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.PluginManager;
import org.bukkit.scheduler.BukkitScheduler;
import org.junit.Before;
import org.junit.Test;

import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static xyz.laur.duels.GameSession.GameMap.FORT_ROYALE;

public class JoinCommandTest {
    private JoinCommand command;
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;

    @Before
    public void setUp() {
        sessionManager = new SessionManager();
        invisibilityManager = mock(InvisibilityManager.class);
        Plugin plugin = mock(Plugin.class);
        when(plugin.getLogger()).thenReturn(Logger.getGlobal());

        Server server = mock(Server.class);
        PluginManager pluginManager = mock(PluginManager.class);
        BukkitScheduler scheduler = mock(BukkitScheduler.class);
        when(server.getPluginManager()).thenReturn(pluginManager);
        when(server.getScheduler()).thenReturn(scheduler);
        when(plugin.getServer()).thenReturn(server);

        World world = mock(World.class);

        command = new JoinCommand(sessionManager, invisibilityManager, plugin, world);
    }

    @Test
    public void testJoinSession() {
        PlayerInventory inventory = mock(PlayerInventory.class);
        Player sender1 = mock(Player.class);
        when(sender1.getInventory()).thenReturn(inventory);

        assertTrue(command.onCommand(sender1, null, "join",
                new String[]{"abc123", "0.334", "fort_royale"}));
        GameSession session = sessionManager.getByName("abc123");
        assertTrue(session.hasPlayer(sender1));
        assertEquals(FORT_ROYALE, session.getMap());

        Player sender2 = mock(Player.class);
        when(sender2.getInventory()).thenReturn(inventory);

        assertTrue(command.onCommand(sender2, null, "join",
                new String[]{"abc123", "0.334", "fort_royale"}));
        assertTrue(session.hasPlayer(sender2));
    }
}
