package xyz.laur.duels;

import com.comphenix.packetwrapper.WrapperPlayServerNamedEntitySpawn;
import com.comphenix.packetwrapper.WrapperPlayServerPlayerInfo;
import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.ProtocolLibrary;
import com.comphenix.protocol.events.PacketAdapter;
import com.comphenix.protocol.events.PacketEvent;
import com.comphenix.protocol.wrappers.EnumWrappers;
import com.comphenix.protocol.wrappers.PlayerInfoData;
import com.comphenix.protocol.wrappers.WrappedGameProfile;
import com.comphenix.protocol.wrappers.WrappedSignedProperty;
import com.google.common.collect.Multimap;
import com.google.gson.Gson;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;

import static com.comphenix.protocol.wrappers.EnumWrappers.PlayerInfoAction.ADD_PLAYER;

public class SkinChanger extends PacketAdapter {
    public static class Skin {
        private final String value;
        private final String signature;

        private Skin(String value, String signature) {
            this.value = value;
            this.signature = signature;
        }
    }

    public static final Skin[] SKINS;

    static {
        InputStream stream = SkinChanger.class.getClassLoader().getResourceAsStream("skin_textures.json");
        assert stream != null;
        SKINS = new Gson().fromJson(new InputStreamReader(stream), Skin[].class);
    }

    private final Random random = new Random();
    private final Map<UUID, Skin> playerSkins = new HashMap<>();

    public SkinChanger(Plugin plugin) {
        super(plugin, PacketType.Play.Server.PLAYER_INFO);

        ProtocolLibrary.getProtocolManager().addPacketListener(this);
    }

    public void changeSkin(Player player) {
        playerSkins.put(player.getUniqueId(), randomSkin());

        sendAddPlayerPacket(player);
        sendPlayerSpawnPacket(player);
    }

    private void sendPlayerSpawnPacket(Player player) {
        WrapperPlayServerNamedEntitySpawn packet = new WrapperPlayServerNamedEntitySpawn();
        packet.setEntityID(player.getEntityId());
        packet.setPlayerUUID(player.getUniqueId());

        Location location = player.getLocation();
        packet.setX(location.getX());
        packet.setY(location.getY());
        packet.setZ(location.getZ());
        packet.setYaw(location.getYaw());
        packet.setPitch(location.getPitch());
        packet.setCurrentItem(player.getItemInHand().getType().getId());

        player.getWorld().getPlayers()
                .stream()
                .filter(p -> p.canSee(player) && isTracked(p, player) && !p.getUniqueId().equals(player.getUniqueId()))
                .forEach(packet::sendPacket);
    }

    private void sendAddPlayerPacket(Player player) {
        WrapperPlayServerPlayerInfo packet = new WrapperPlayServerPlayerInfo();
        packet.setAction(ADD_PLAYER);
        WrappedGameProfile profile = new WrappedGameProfile(player.getUniqueId(), player.getName());
        Skin skin = getPlayerSkin(player);
        profile.getProperties().put("textures",
                new WrappedSignedProperty("textures", skin.value, skin.signature));

        int ping = getPing(player);
        EnumWrappers.NativeGameMode gameMode = EnumWrappers.NativeGameMode.fromBukkit(player.getGameMode());

        List<PlayerInfoData> data = new ArrayList<>();
        data.add(new PlayerInfoData(profile, ping, gameMode, null));
        packet.setData(data);

        player.getWorld().getPlayers()
                .stream()
                .filter(p -> p.canSee(player) && isTracked(p, player) && !p.getUniqueId().equals(player.getUniqueId()))
                .forEach(packet::sendPacket);
    }

    @Override
    public void onPacketSending(PacketEvent event) {
        WrapperPlayServerPlayerInfo packet = new WrapperPlayServerPlayerInfo(event.getPacket());

        if (packet.getAction() == ADD_PLAYER) {
            for (PlayerInfoData info : packet.getData()) {
                Multimap<String, WrappedSignedProperty> properties = info.getProfile().getProperties();

                if (!properties.containsKey("textures") &&
                        !event.getPlayer().getUniqueId().equals(info.getProfile().getUUID())) {
                    Player player = plugin.getServer().getPlayer(info.getProfile().getUUID());
                    Skin skin = getPlayerSkin(player);
                    properties.put("textures",
                            new WrappedSignedProperty("textures", skin.value, skin.signature));
                }
            }
        }
    }

    private Skin randomSkin() {
        return SKINS[random.nextInt(SKINS.length)];
    }

    private Skin getPlayerSkin(Player player) {
        if (!playerSkins.containsKey(player.getUniqueId())) {
            playerSkins.put(player.getUniqueId(), randomSkin());
        }

        return playerSkins.get(player.getUniqueId());
    }

    private int getPing(Player target) {
        Class<?> craftPlayer = target.getClass();
        try {
            Method getHandle = craftPlayer.getMethod("getHandle", (Class[]) null);
            Object entityPlayer = getHandle.invoke(target);
            Field pingField = entityPlayer.getClass().getField("ping");
            return (int) pingField.get(entityPlayer);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    // Whether a can see b
    private boolean isTracked(Player a, Player b) {
        Class<? extends World> worldClass = b.getWorld().getClass();
        try {
            Field worldServerField = worldClass.getDeclaredField("world");
            worldServerField.setAccessible(true);
            Object worldServer = worldServerField.get(b.getWorld());

            Field trackerField = worldServer.getClass().getField("tracker");
            Object tracker = trackerField.get(worldServer);

            Field trackedEntitiesField = tracker.getClass().getField("trackedEntities");
            Object trackedEntities = trackedEntitiesField.get(tracker);

            Method getTrackerEntry = trackedEntities.getClass().getMethod("get", int.class);
            Object trackerEntry = getTrackerEntry.invoke(trackedEntities, b.getEntityId());

            Field trackedPlayersField = trackerEntry.getClass().getField("trackedPlayers");
            Set<Object> trackedPlayers = (Set<Object>) trackedPlayersField.get(trackerEntry);

            Method getPlayerHandle = a.getClass().getMethod("getHandle");
            Object aHandle = getPlayerHandle.invoke(a);

            return trackedPlayers.contains(aHandle);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
}
