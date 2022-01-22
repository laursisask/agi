package xyz.laur.duels;

import org.bukkit.Server;
import org.bukkit.entity.Player;

public class InvisibilityManager {
    private final SessionManager sessionManager;
    private final Server server;

    public InvisibilityManager(SessionManager sessionManager, Server server) {
        this.sessionManager = sessionManager;
        this.server = server;
    }

    public void update() {
        for (Player a : server.getOnlinePlayers()) {
            for (Player b : server.getOnlinePlayers()) {
                GameSession sessionA = sessionManager.getPlayerSession(a);
                GameSession sessionB = sessionManager.getPlayerSession(b);
                boolean shouldSee = sessionA == null || sessionA.equals(sessionB);

                if (a.canSee(b) != shouldSee) {
                    if (shouldSee) {
                        a.showPlayer(b);
                    } else {
                        a.hidePlayer(b);
                    }
                }
            }
        }
    }
}
