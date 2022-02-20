package xyz.laur.duels;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.Server;
import org.bukkit.World;
import org.bukkit.entity.*;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityShootBowEvent;
import org.bukkit.event.player.PlayerFishEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.PluginManager;
import org.bukkit.scheduler.BukkitScheduler;
import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static org.junit.Assert.*;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.*;
import static xyz.laur.duels.ClassicGameSession.GameMap.ARENA;
import static xyz.laur.duels.ClassicGameSession.GameMap.BACKWOODS;
import static xyz.laur.duels.ClassicGameSession.MAX_DURATION;
import static xyz.laur.duels.TestHelper.mockServer;

public class ClassicGameSessionTest {
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;
    private ClassicGameSession session;
    private World world;
    private BukkitScheduler scheduler;
    private SkinChanger skinChanger;
    private Plugin plugin;
    private Map<ClassicGameSession.GameMap, MapBarrier> barriers;

    @Before
    public void setUp() {
        mockServer();

        sessionManager = new SessionManager();
        invisibilityManager = mock(InvisibilityManager.class);
        this.plugin = mock(Plugin.class);
        when(plugin.getLogger()).thenReturn(Logger.getGlobal());

        Server server = mock(Server.class);
        PluginManager pluginManager = mock(PluginManager.class);
        scheduler = mock(BukkitScheduler.class);
        when(server.getPluginManager()).thenReturn(pluginManager);
        when(server.getScheduler()).thenReturn(scheduler);
        when(plugin.getServer()).thenReturn(server);

        world = mock(World.class);

        skinChanger = mock(SkinChanger.class);

        barriers = new HashMap<>();
        for (ClassicGameSession.GameMap map : ClassicGameSession.GameMap.values()) {
            barriers.put(map, mock(MapBarrier.class));
        }

        session = new ClassicGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin, ARENA,
                true, 1, barriers);
    }

    @Test
    public void testGameMainLifecycle() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        verify(invisibilityManager, times(1)).update();

        assertEquals(GameState.WAITING_FOR_PLAYERS, session.getState());
        assertTrue(session.hasPlayer(player1));
        verify(player1, never()).teleport(any(Location.class));

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());
        assertTrue(session.hasPlayer(player1));
        assertTrue(session.hasPlayer(player2));
        verify(player1, times(1)).sendMessage("Game started");
        verify(player2, times(1)).sendMessage("Game started");

        verify(invisibilityManager, times(2)).update();

        verify(player1, times(1)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));

        verify(player1.getInventory(), times(1)).clear();
        verify(player2.getInventory(), times(1)).clear();

        // Random entity dying
        Sheep sheep = mock(Sheep.class);
        session.onEntityDamage(createDamageEvent(sheep, 10));
        assertEquals(GameState.PLAYING, session.getState());

        // Player in another game dying
        Player unrelatedPlayer = mock(Player.class);
        session.onEntityDamage(createDamageEvent(unrelatedPlayer, 10));
        assertEquals(GameState.PLAYING, session.getState());

        // Player in the same game taking damage but not dying
        when(player2.getHealth()).thenReturn(7D);
        EntityDamageEvent event1 = createDamageEvent(player2, 5);
        session.onEntityDamage(event1);
        assertFalse(event1.isCancelled());
        assertEquals(GameState.PLAYING, session.getState());

        // Player in the same game dying
        when(player1.getHealth()).thenReturn(8D);
        EntityDamageEvent event2 = createDamageEvent(player1, 11);
        session.onEntityDamage(event2);
        assertTrue(event2.isCancelled());
        assertEquals(GameState.ENDED, session.getState());

        verify(player1, times(1)).sendMessage("You lost");
        verify(player1, never()).sendMessage(startsWith("metadata:win"));

        verify(player2, times(1)).sendMessage("You won");
        verify(player2, times(1)).sendMessage("metadata:win:1.00000");
        verify(invisibilityManager, times(3)).update();
    }

    private EntityDamageEvent createDamageEvent(Entity damagee, double damage) {
        Map<EntityDamageEvent.DamageModifier, Double> modifiers = new HashMap<>();
        modifiers.put(EntityDamageEvent.DamageModifier.BASE, damage);
        Map<EntityDamageEvent.DamageModifier, Function<? super Double, Double>> modifierFunctions = new HashMap<>();
        modifierFunctions.put(EntityDamageEvent.DamageModifier.BASE, Functions.constant(0D));

        return new EntityDamageEvent(damagee, EntityDamageEvent.DamageCause.ENTITY_ATTACK, modifiers,
                modifierFunctions);
    }

    @Test
    public void testPlayerQuits() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());

        session.onQuit(new PlayerQuitEvent(player2, null));
        assertEquals(GameState.ENDED, session.getState());

        verify(player1, times(1)).sendMessage("You won");
        verify(player1, times(1)).sendMessage("metadata:win:1.00000");

        verify(player2, times(1)).sendMessage("You lost");
        verify(player2, never()).sendMessage(startsWith("metadata:win"));
    }

    @Test
    public void testTimeout() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());

        ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
        verify(scheduler, times(1))
                .scheduleSyncDelayedTask(any(Plugin.class), task.capture(), eq(MAX_DURATION));

        task.getValue().run();
        assertEquals(GameState.ENDED, session.getState());
        verify(player1, times(1)).sendMessage("You lost");
        verify(player1, never()).sendMessage(startsWith("metadata:win"));

        verify(player2, times(1)).sendMessage("You lost");
        verify(player2, never()).sendMessage(startsWith("metadata:win"));
    }

    @Test
    public void testDirectAttack() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getDamager()).thenReturn(player1);
        when(damageEvent.getEntity()).thenReturn(player2);

        session.onPlayerDamage(damageEvent);

        verify(player1, times(1)).sendMessage("metadata:hits_done:3.50000");
        verify(player2, times(1)).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testBowAttack() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getEntity()).thenReturn(player2);

        Arrow arrow = mock(Arrow.class);
        when(arrow.getShooter()).thenReturn(player1);
        when(damageEvent.getDamager()).thenReturn(arrow);

        session.onPlayerDamage(damageEvent);

        verify(damageEvent, never()).setCancelled(anyBoolean());

        verify(player1, times(1)).sendMessage("metadata:hits_done:3.50000");
        verify(player2, times(1)).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testBowAttackByUnrelatedArrow() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getEntity()).thenReturn(player2);

        Arrow arrow = mock(Arrow.class);
        when(arrow.getType()).thenReturn(EntityType.ARROW);
        when(arrow.getShooter()).thenReturn(mock(Player.class));
        when(damageEvent.getDamager()).thenReturn(arrow);

        session.onPlayerDamage(damageEvent);

        verify(damageEvent, times(1)).setCancelled(eq(true));

        verify(player1, never()).sendMessage("metadata:hits_done:3.50000");
        verify(player2, never()).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testFishingHookAttack() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(0D);
        when(damageEvent.getEntity()).thenReturn(player2);

        FishHook hook = mock(FishHook.class);
        when(hook.getType()).thenReturn(EntityType.FISHING_HOOK);
        when(hook.getShooter()).thenReturn(player1);
        when(damageEvent.getDamager()).thenReturn(hook);

        session.onPlayerDamage(damageEvent);

        verify(damageEvent, never()).setCancelled(anyBoolean());

        verify(player1, never()).sendMessage("metadata:hits_done:0.00000");
        verify(player2, never()).sendMessage("metadata:hits_received:0.00000");

        verify(player1, times(1)).sendMessage("metadata:fishing_hook_hit:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_hook_hit:1.00000");
    }

    @Test
    public void testShootArrowMetadata() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Arrow arrow = mock(Arrow.class);
        EntityShootBowEvent event = new EntityShootBowEvent(player1, new ItemStack(Material.BOW), arrow, 0.5F);
        session.onBowShoot(event);

        verify(player1, times(1)).sendMessage("metadata:shoot_arrow:0.50000");
        verify(player2, never()).sendMessage("metadata:shoot_arrow:0.50000");
    }

    @Test
    public void testShootArrowUnrelatedPlayer() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Player player3 = createMockPlayer();

        Arrow arrow = mock(Arrow.class);
        EntityShootBowEvent event = new EntityShootBowEvent(player3, new ItemStack(Material.BOW), arrow, 0.5F);
        session.onBowShoot(event);

        verify(player3, never()).sendMessage(anyString());
    }

    @Test
    public void testFishingRodMetadata() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        session.onPlayerFish(new PlayerFishEvent(player1, null, null, PlayerFishEvent.State.FISHING));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");

        session.onPlayerFish(new PlayerFishEvent(player1, null, null, PlayerFishEvent.State.IN_GROUND));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");

        session.onPlayerFish(new PlayerFishEvent(player1, null, null, PlayerFishEvent.State.CAUGHT_FISH));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");

        session.onPlayerFish(new PlayerFishEvent(player1, null, null, PlayerFishEvent.State.FAILED_ATTEMPT));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");

        session.onPlayerFish(new PlayerFishEvent(player1, null, null, PlayerFishEvent.State.CAUGHT_ENTITY));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");

        Player player3 = createMockPlayer();
        session.onPlayerFish(new PlayerFishEvent(player3, null, null, PlayerFishEvent.State.FISHING));
        verify(player1, times(1)).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player2, never()).sendMessage("metadata:fishing_rod_used:1.00000");
        verify(player3, never()).sendMessage(anyString());
    }

    @Test
    public void testRandomTeleport() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());

        verify(player1, atLeast(1)).teleport(any(Location.class));
        verify(player2, atLeast(1)).teleport(any(Location.class));

        ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
        verify(scheduler, times(1)).scheduleSyncRepeatingTask(any(), task.capture(), eq(40L), eq(1L));

        when(player1.getLocation()).thenReturn(new Location(world, 220, 33, 25));
        when(player2.getLocation()).thenReturn(new Location(world, 222, 33.5, 24));

        ArgumentCaptor<Location> teleportLocs1 = ArgumentCaptor.forClass(Location.class);
        ArgumentCaptor<Location> teleportLocs2 = ArgumentCaptor.forClass(Location.class);

        verify(player1, atLeast(1)).teleport(teleportLocs1.capture());
        verify(player2, atLeast(1)).teleport(teleportLocs2.capture());

        for (int i = 0; i < 4000; i++) {
            task.getValue().run();
        }

        verify(player1, atLeast(2)).teleport(any(Location.class));
        verify(player2, atLeast(2)).teleport(any(Location.class));
    }

    @Test
    public void testSpawnLocationsFullDistance() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        ArgumentCaptor<Location> player1Loc = ArgumentCaptor.forClass(Location.class);
        verify(player1, times(1)).teleport(player1Loc.capture());

        ArgumentCaptor<Location> player2Loc = ArgumentCaptor.forClass(Location.class);
        verify(player2, times(1)).teleport(player2Loc.capture());

        Location potentialSpawn1 = ARENA.getSpawnLocation1().clone();
        potentialSpawn1.setWorld(world);

        Location potentialSpawn2 = ARENA.getSpawnLocation2().clone();
        potentialSpawn2.setWorld(world);

        assertTrue((player1Loc.getValue().equals(potentialSpawn1) && player2Loc.getValue().equals(potentialSpawn2)) ||
                (player1Loc.getValue().equals(potentialSpawn2) && player2Loc.getValue().equals(potentialSpawn1)));
    }

    @Test
    public void testSpawnLocationsHalfDistanceX() {
        session = new ClassicGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin, BACKWOODS,
                true, 0.5F, barriers);

        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        ArgumentCaptor<Location> player1Loc = ArgumentCaptor.forClass(Location.class);
        verify(player1, times(1)).teleport(player1Loc.capture());

        ArgumentCaptor<Location> player2Loc = ArgumentCaptor.forClass(Location.class);
        verify(player2, times(1)).teleport(player2Loc.capture());

        Location potentialSpawn1 = new Location(world, 22, 68, -226.5, -90, 0);
        Location potentialSpawn2 = new Location(world, 57, 68, -226.5, 90, 0);

        assertTrue((player1Loc.getValue().equals(potentialSpawn1) && player2Loc.getValue().equals(potentialSpawn2)) ||
                (player1Loc.getValue().equals(potentialSpawn2) && player2Loc.getValue().equals(potentialSpawn1)));
    }

    @Test
    public void testSpawnLocationsHalfDistanceZ() {
        session = new ClassicGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin, ARENA,
                true, 0.5F, barriers);

        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        ArgumentCaptor<Location> player1Loc = ArgumentCaptor.forClass(Location.class);
        verify(player1, times(1)).teleport(player1Loc.capture());

        ArgumentCaptor<Location> player2Loc = ArgumentCaptor.forClass(Location.class);
        verify(player2, times(1)).teleport(player2Loc.capture());

        Location potentialSpawn1 = new Location(world, 43.5, 71, -78.5, 0, 0);
        Location potentialSpawn2 = new Location(world, 43.5, 71, -44.5, -180, 0);

        assertTrue((player1Loc.getValue().equals(potentialSpawn1) && player2Loc.getValue().equals(potentialSpawn2)) ||
                (player1Loc.getValue().equals(potentialSpawn2) && player2Loc.getValue().equals(potentialSpawn1)));
    }

    protected Player createMockPlayer() {
        Player player = mock(Player.class);
        PlayerInventory inventory = mock(PlayerInventory.class);
        when(player.getInventory()).thenReturn(inventory);

        return player;
    }
}
