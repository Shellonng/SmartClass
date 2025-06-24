package com.education;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * AI赋能教育管理与学习辅助平台 - 主启动类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@SpringBootApplication(exclude = {
    org.springframework.boot.actuate.autoconfigure.metrics.MetricsAutoConfiguration.class
})
@EnableCaching
@EnableAsync
@EnableScheduling
@EnableTransactionManagement
public class EducationApplication {

    public static void main(String[] args) {
        SpringApplication.run(EducationApplication.class, args);
        System.out.println("");
        System.out.println("  ______ _____  _    _  _____       _______ _____ ____  _   _ ");
        System.out.println(" |  ____|  __ \\| |  | |/ ____|   /\\|__   __|_   _/ __ \\| \\ | |");
        System.out.println(" | |__  | |  | | |  | | |       /  \\  | |    | || |  | |  \\| |");
        System.out.println(" |  __| | |  | | |  | | |      / /\\ \\ | |    | || |  | | . ` |");
        System.out.println(" | |____| |__| | |__| | |____ / ____ \\| |   _| || |__| | |\\  |");
        System.out.println(" |______|_____/ \\____/ \\_____/_/    \\_\\_|  |_____\\____/|_| \\_|");
        System.out.println("");
        System.out.println(" :: AI赋能教育管理与学习辅助平台 :: (v1.0.0)");
        System.out.println("");
        System.out.println(" 🚀 应用启动成功！");
        System.out.println(" 📖 API文档地址: http://localhost:8080/doc.html");
        System.out.println(" 🔧 监控地址: http://localhost:8080/actuator");
        System.out.println("");
    }
}