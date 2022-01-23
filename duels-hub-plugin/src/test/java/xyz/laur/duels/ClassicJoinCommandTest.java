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
import static xyz.laur.duels.ClassicGameSession.GameMap.ARENA;
import static xyz.laur.duels.TestHelper.mockServer;

public class ClassicJoinCommandTest {
    private ClassicJoinCommand command;
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;

    @Before
    public void setUp() {
        mockServer();

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

        SkinChanger skinChanger = mock(SkinChanger.class);

        command = new ClassicJoinCommand(sessionManager, invisibilityManager, skinChanger, plugin, world);
    }

    @Test
    public void testJoinSession() {
        PlayerInventory inventory = mock(PlayerInventory.class);
        Player sender1 = mock(Player.class);
        when(sender1.getInventory()).thenReturn(inventory);

        assertTrue(command.onCommand(sender1, null, "classic", new String[]{"abc123", "arena", "true"}));
        ClassicGameSession session = (ClassicGameSession) sessionManager.getByName("abc123");
        assertTrue(session.hasPlayer(sender1));
        assertEquals(ARENA, session.getMap());
        assertTrue(session.hasRandomTeleport());

        Player sender2 = mock(Player.class);
        when(sender2.getInventory()).thenReturn(inventory);

        assertTrue(command.onCommand(sender2, null, "classic", new String[]{"abc123", "arena", "true"}));
        assertTrue(session.hasPlayer(sender2));
    }
}
