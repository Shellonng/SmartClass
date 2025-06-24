# 后端架构设计文档

## 一、项目结构设计

### 1.1 包结构

```
com.education/
├── EducationApplication.java        # 启动类
├── config/                          # 配置类
│   ├── SecurityConfig.java          # 安全配置
│   ├── RedisConfig.java             # Redis配置
│   ├── MyBatisPlusConfig.java       # MyBatis-Plus配置
│   ├── CorsConfig.java              # 跨域配置
│   └── SwaggerConfig.java           # API文档配置
├── controller/                      # 控制器层
│   ├── auth/                        # 认证相关
│   │   └── AuthController.java
│   ├── teacher/                     # 教师端接口
│   │   ├── ClassController.java
│   │   ├── StudentController.java
│   │   ├── CourseController.java
│   │   ├── TaskController.java
│   │   ├── GradeController.java
│   │   ├── ResourceController.java
│   │   ├── KnowledgeController.java
│   │   └── AIController.java
│   ├── student/                     # 学生端接口
│   │   ├── CourseController.java
│   │   ├── TaskController.java
│   │   ├── GradeController.java
│   │   ├── ResourceController.java
│   │   └── AILearningController.java
│   └── common/                      # 公共接口
│       ├── FileController.java
│       └── UserController.java
├── service/                         # 服务层
│   ├── auth/
│   │   ├── AuthService.java
│   │   └── impl/
│   │       └── AuthServiceImpl.java
│   ├── teacher/
│   │   ├── ClassService.java
│   │   ├── StudentService.java
│   │   ├── CourseService.java
│   │   ├── TaskService.java
│   │   ├── GradeService.java
│   │   ├── ResourceService.java
│   │   ├── KnowledgeService.java
│   │   ├── AIService.java
│   │   └── impl/
│   ├── student/
│   │   ├── StudentCourseService.java
│   │   ├── StudentTaskService.java
│   │   ├── StudentGradeService.java
│   │   ├── StudentResourceService.java
│   │   └── impl/
│   └── common/
│       ├── FileService.java
│       ├── UserService.java
│       ├── EmailService.java
│       ├── RedisService.java
│       └── impl/
├── mapper/                          # 数据访问层
│   ├── UserMapper.java
│   ├── StudentMapper.java
│   ├── TeacherMapper.java
│   ├── ClassMapper.java
│   ├── CourseMapper.java
│   ├── TaskMapper.java
│   ├── GradeMapper.java
│   ├── ResourceMapper.java
│   ├── KnowledgeGraphMapper.java
│   ├── QuestionBankMapper.java
│   ├── SubmissionMapper.java
│   └── AIFeatureMapper.java
├── entity/                          # 实体类
│   ├── User.java
│   ├── Student.java
│   ├── Teacher.java
│   ├── Class.java
│   ├── Course.java
│   ├── Task.java
│   ├── Grade.java
│   ├── Resource.java
│   ├── KnowledgeGraph.java
│   ├── QuestionBank.java
│   ├── Submission.java
│   └── AIFeature.java
├── dto/                             # 数据传输对象
│   ├── request/
│   │   ├── LoginRequest.java
│   │   ├── ClassCreateRequest.java
│   │   ├── TaskCreateRequest.java
│   │   └── ...
│   ├── response/
│   │   ├── LoginResponse.java
│   │   ├── ClassResponse.java
│   │   ├── TaskResponse.java
│   │   └── ...
│   └── common/
│       ├── PageRequest.java
│       ├── PageResponse.java
│       └── Result.java
├── utils/                           # 工具类
│   ├── JwtUtils.java               # JWT工具
│   ├── PasswordUtils.java          # 密码工具
│   ├── FileUtils.java              # 文件工具
│   ├── DateUtils.java              # 日期工具
│   ├── ValidationUtils.java        # 验证工具
│   └── RedisUtils.java             # Redis工具
├── exception/                       # 异常处理
│   ├── GlobalExceptionHandler.java # 全局异常处理
│   ├── BusinessException.java      # 业务异常
│   └── ErrorCode.java              # 错误码定义
├── security/                        # 安全相关
│   ├── JwtAuthenticationFilter.java
│   ├── JwtAuthenticationEntryPoint.java
│   └── UserDetailsServiceImpl.java
└── aspect/                          # 切面
    ├── LogAspect.java              # 日志切面
    ├── CacheAspect.java            # 缓存切面
    └── PermissionAspect.java       # 权限切面
```

## 二、核心业务模块设计

### 2.1 认证模块 (Auth)

#### AuthController.java

```java
@RestController
@RequestMapping("/api/auth")
@Api(tags = "认证管理")
public class AuthController {
    
    @Autowired
    private AuthService authService;
    
    /**
     * 用户登录
     */
    @PostMapping("/login")
    @ApiOperation("用户登录")
    public Result<LoginResponse> login(@RequestBody @Valid LoginRequest request) {
        LoginResponse response = authService.login(request);
        return Result.success(response);
    }
    
    /**
     * 用户注册
     */
    @PostMapping("/register")
    @ApiOperation("用户注册")
    public Result<Void> register(@RequestBody @Valid RegisterRequest request) {
        authService.register(request);
        return Result.success();
    }
    
    /**
     * 获取验证码
     */
    @GetMapping("/captcha")
    @ApiOperation("获取验证码")
    public Result<CaptchaResponse> getCaptcha() {
        CaptchaResponse response = authService.generateCaptcha();
        return Result.success(response);
    }
    
    /**
     * 刷新Token
     */
    @PostMapping("/refresh")
    @ApiOperation("刷新Token")
    public Result<TokenResponse> refreshToken(@RequestBody RefreshTokenRequest request) {
        TokenResponse response = authService.refreshToken(request.getRefreshToken());
        return Result.success(response);
    }
    
    /**
     * 用户登出
     */
    @PostMapping("/logout")
    @ApiOperation("用户登出")
    @PreAuthorize("hasRole('USER')")
    public Result<Void> logout(HttpServletRequest request) {
        String token = JwtUtils.getTokenFromRequest(request);
        authService.logout(token);
        return Result.success();
    }
}
```

#### AuthService接口设计

```java
public interface AuthService {
    
    /**
     * 用户登录
     */
    LoginResponse login(LoginRequest request);
    
    /**
     * 用户注册
     */
    void register(RegisterRequest request);
    
    /**
     * 生成验证码
     */
    CaptchaResponse generateCaptcha();
    
    /**
     * 刷新Token
     */
    TokenResponse refreshToken(String refreshToken);
    
    /**
     * 用户登出
     */
    void logout(String token);
    
    /**
     * 验证Token
     */
    boolean validateToken(String token);
}
```

### 2.2 教师端 - 班级管理模块

#### ClassController.java

```java
@RestController
@RequestMapping("/api/teacher/classes")
@Api(tags = "班级管理")
@PreAuthorize("hasRole('TEACHER')")
public class ClassController {
    
    @Autowired
    private ClassService classService;
    
    /**
     * 获取班级列表
     */
    @GetMapping
    @ApiOperation("获取班级列表")
    public Result<PageResponse<ClassResponse>> getClasses(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) String semester) {
        
        PageRequest pageRequest = new PageRequest(page, size);
        ClassQueryRequest queryRequest = new ClassQueryRequest(keyword, semester);
        PageResponse<ClassResponse> response = classService.getClasses(pageRequest, queryRequest);
        return Result.success(response);
    }
    
    /**
     * 获取班级详情
     */
    @GetMapping("/{id}")
    @ApiOperation("获取班级详情")
    public Result<ClassDetailResponse> getClassDetail(@PathVariable Long id) {
        ClassDetailResponse response = classService.getClassDetail(id);
        return Result.success(response);
    }
    
    /**
     * 创建班级
     */
    @PostMapping
    @ApiOperation("创建班级")
    public Result<Void> createClass(@RequestBody @Valid ClassCreateRequest request) {
        classService.createClass(request);
        return Result.success();
    }
    
    /**
     * 更新班级
     */
    @PutMapping("/{id}")
    @ApiOperation("更新班级")
    public Result<Void> updateClass(@PathVariable Long id, 
                                   @RequestBody @Valid ClassUpdateRequest request) {
        classService.updateClass(id, request);
        return Result.success();
    }
    
    /**
     * 删除班级
     */
    @DeleteMapping("/{id}")
    @ApiOperation("删除班级")
    public Result<Void> deleteClass(@PathVariable Long id) {
        classService.deleteClass(id);
        return Result.success();
    }
    
    /**
     * 获取班级学生列表
     */
    @GetMapping("/{id}/students")
    @ApiOperation("获取班级学生列表")
    public Result<PageResponse<StudentResponse>> getClassStudents(
            @PathVariable Long id,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        
        PageRequest pageRequest = new PageRequest(page, size);
        PageResponse<StudentResponse> response = classService.getClassStudents(id, pageRequest);
        return Result.success(response);
    }
    
    /**
     * 添加学生到班级
     */
    @PostMapping("/{id}/students")
    @ApiOperation("添加学生到班级")
    public Result<Void> addStudentsToClass(@PathVariable Long id, 
                                          @RequestBody @Valid AddStudentsRequest request) {
        classService.addStudentsToClass(id, request.getStudentIds());
        return Result.success();
    }
    
    /**
     * 从班级移除学生
     */
    @DeleteMapping("/{id}/students/{studentId}")
    @ApiOperation("从班级移除学生")
    public Result<Void> removeStudentFromClass(@PathVariable Long id, 
                                              @PathVariable Long studentId) {
        classService.removeStudentFromClass(id, studentId);
        return Result.success();
    }
    
    /**
     * 批量导入学生
     */
    @PostMapping("/{id}/students/import")
    @ApiOperation("批量导入学生")
    public Result<ImportResult> importStudents(@PathVariable Long id, 
                                              @RequestParam("file") MultipartFile file) {
        ImportResult result = classService.importStudents(id, file);
        return Result.success(result);
    }
    
    /**
     * 导出班级学生
     */
    @GetMapping("/{id}/students/export")
    @ApiOperation("导出班级学生")
    public void exportStudents(@PathVariable Long id, HttpServletResponse response) {
        classService.exportStudents(id, response);
    }
}
```

### 2.3 教师端 - 成绩管理模块

#### GradeController.java

```java
@RestController
@RequestMapping("/api/teacher/grades")
@Api(tags = "成绩管理")
@PreAuthorize("hasRole('TEACHER')")
public class GradeController {
    
    @Autowired
    private GradeService gradeService;
    
    /**
     * 获取成绩统计
     */
    @GetMapping("/statistics")
    @ApiOperation("获取成绩统计")
    @Cacheable(value = "grade:statistics", key = "#courseId + ':' + #classId")
    public Result<GradeStatisticsResponse> getGradeStatistics(
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) Long classId,
            @RequestParam(required = false) String timeRange) {
        
        GradeStatisticsRequest request = new GradeStatisticsRequest(courseId, classId, timeRange);
        GradeStatisticsResponse response = gradeService.getGradeStatistics(request);
        return Result.success(response);
    }
    
    /**
     * 获取成绩列表
     */
    @GetMapping
    @ApiOperation("获取成绩列表")
    public Result<PageResponse<GradeResponse>> getGrades(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) Long classId,
            @RequestParam(required = false) Long taskId) {
        
        PageRequest pageRequest = new PageRequest(page, size);
        GradeQueryRequest queryRequest = new GradeQueryRequest(courseId, classId, taskId);
        PageResponse<GradeResponse> response = gradeService.getGrades(pageRequest, queryRequest);
        return Result.success(response);
    }
    
    /**
     * 获取学生成绩详情
     */
    @GetMapping("/student/{studentId}")
    @ApiOperation("获取学生成绩详情")
    public Result<StudentGradeDetailResponse> getStudentGradeDetail(
            @PathVariable Long studentId,
            @RequestParam(required = false) Long courseId) {
        
        StudentGradeDetailResponse response = gradeService.getStudentGradeDetail(studentId, courseId);
        return Result.success(response);
    }
    
    /**
     * 批量录入成绩
     */
    @PostMapping("/batch")
    @ApiOperation("批量录入成绩")
    public Result<Void> batchCreateGrades(@RequestBody @Valid BatchGradeCreateRequest request) {
        gradeService.batchCreateGrades(request);
        return Result.success();
    }
    
    /**
     * 更新成绩
     */
    @PutMapping("/{id}")
    @ApiOperation("更新成绩")
    public Result<Void> updateGrade(@PathVariable Long id, 
                                   @RequestBody @Valid GradeUpdateRequest request) {
        gradeService.updateGrade(id, request);
        return Result.success();
    }
    
    /**
     * 生成个性化反馈
     */
    @PostMapping("/feedback")
    @ApiOperation("生成个性化反馈")
    public Result<FeedbackResponse> generateFeedback(@RequestBody @Valid FeedbackRequest request) {
        FeedbackResponse response = gradeService.generateFeedback(request);
        return Result.success(response);
    }
    
    /**
     * 导出成绩报表
     */
    @GetMapping("/export")
    @ApiOperation("导出成绩报表")
    public void exportGrades(@RequestParam(required = false) Long courseId,
                           @RequestParam(required = false) Long classId,
                           @RequestParam(defaultValue = "excel") String format,
                           HttpServletResponse response) {
        gradeService.exportGrades(courseId, classId, format, response);
    }
    
    /**
     * AI成绩分析 (预留接口)
     */
    @PostMapping("/ai-analysis")
    @ApiOperation("AI成绩分析")
    public Result<AIAnalysisResponse> aiGradeAnalysis(@RequestBody @Valid AIAnalysisRequest request) {
        // TODO: 接入AI分析服务
        AIAnalysisResponse response = gradeService.aiGradeAnalysis(request);
        return Result.success(response);
    }
}
```

### 2.4 学生端 - 任务管理模块

#### StudentTaskController.java

```java
@RestController
@RequestMapping("/api/student/tasks")
@Api(tags = "学生任务管理")
@PreAuthorize("hasRole('STUDENT')")
public class StudentTaskController {
    
    @Autowired
    private StudentTaskService studentTaskService;
    
    /**
     * 获取学生任务列表
     */
    @GetMapping
    @ApiOperation("获取学生任务列表")
    public Result<PageResponse<StudentTaskResponse>> getTasks(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long courseId) {
        
        Long studentId = SecurityUtils.getCurrentUserId();
        PageRequest pageRequest = new PageRequest(page, size);
        StudentTaskQueryRequest queryRequest = new StudentTaskQueryRequest(status, courseId);
        PageResponse<StudentTaskResponse> response = studentTaskService.getTasks(studentId, pageRequest, queryRequest);
        return Result.success(response);
    }
    
    /**
     * 获取任务详情
     */
    @GetMapping("/{id}")
    @ApiOperation("获取任务详情")
    public Result<TaskDetailResponse> getTaskDetail(@PathVariable Long id) {
        Long studentId = SecurityUtils.getCurrentUserId();
        TaskDetailResponse response = studentTaskService.getTaskDetail(studentId, id);
        return Result.success(response);
    }
    
    /**
     * 提交任务
     */
    @PostMapping("/{id}/submit")
    @ApiOperation("提交任务")
    public Result<Void> submitTask(@PathVariable Long id, 
                                  @RequestBody @Valid TaskSubmissionRequest request) {
        Long studentId = SecurityUtils.getCurrentUserId();
        studentTaskService.submitTask(studentId, id, request);
        return Result.success();
    }
    
    /**
     * 上传任务文件
     */
    @PostMapping("/{id}/upload")
    @ApiOperation("上传任务文件")
    public Result<FileUploadResponse> uploadTaskFile(@PathVariable Long id, 
                                                    @RequestParam("file") MultipartFile file) {
        Long studentId = SecurityUtils.getCurrentUserId();
        FileUploadResponse response = studentTaskService.uploadTaskFile(studentId, id, file);
        return Result.success(response);
    }
    
    /**
     * 获取任务提交历史
     */
    @GetMapping("/{id}/submissions")
    @ApiOperation("获取任务提交历史")
    public Result<List<SubmissionResponse>> getTaskSubmissions(@PathVariable Long id) {
        Long studentId = SecurityUtils.getCurrentUserId();
        List<SubmissionResponse> response = studentTaskService.getTaskSubmissions(studentId, id);
        return Result.success(response);
    }
    
    /**
     * AI任务辅助 (预留接口)
     */
    @PostMapping("/{id}/ai-assist")
    @ApiOperation("AI任务辅助")
    public Result<AIAssistResponse> getAIAssist(@PathVariable Long id, 
                                               @RequestBody @Valid AIAssistRequest request) {
        Long studentId = SecurityUtils.getCurrentUserId();
        // TODO: 接入AI辅助服务
        AIAssistResponse response = studentTaskService.getAIAssist(studentId, id, request);
        return Result.success(response);
    }
}
```

## 三、统一响应体设计

### 3.1 Result统一响应类

```java
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Result<T> {
    
    private Integer code;
    private String message;
    private T data;
    private Long timestamp;
    
    public static <T> Result<T> success() {
        return new Result<>(200, "操作成功", null, System.currentTimeMillis());
    }
    
    public static <T> Result<T> success(T data) {
        return new Result<>(200, "操作成功", data, System.currentTimeMillis());
    }
    
    public static <T> Result<T> success(String message, T data) {
        return new Result<>(200, message, data, System.currentTimeMillis());
    }
    
    public static <T> Result<T> error(String message) {
        return new Result<>(500, message, null, System.currentTimeMillis());
    }
    
    public static <T> Result<T> error(Integer code, String message) {
        return new Result<>(code, message, null, System.currentTimeMillis());
    }
}
```

### 3.2 分页响应类

```java
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PageResponse<T> {
    
    private List<T> records;
    private Long total;
    private Integer page;
    private Integer size;
    private Integer pages;
    
    public static <T> PageResponse<T> of(IPage<T> page) {
        return new PageResponse<>(
            page.getRecords(),
            page.getTotal(),
            (int) page.getCurrent(),
            (int) page.getSize(),
            (int) page.getPages()
        );
    }
}
```

## 四、安全配置

### 4.1 JWT配置

```java
@Component
public class JwtUtils {
    
    @Value("${jwt.secret}")
    private String secret;
    
    @Value("${jwt.expiration}")
    private Long expiration;
    
    /**
     * 生成Token
     */
    public String generateToken(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("role", userDetails.getAuthorities().iterator().next().getAuthority());
        return createToken(claims, userDetails.getUsername());
    }
    
    /**
     * 创建Token
     */
    private String createToken(Map<String, Object> claims, String subject) {
        return Jwts.builder()
                .setClaims(claims)
                .setSubject(subject)
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + expiration * 1000))
                .signWith(SignatureAlgorithm.HS512, secret)
                .compact();
    }
    
    /**
     * 验证Token
     */
    public Boolean validateToken(String token, UserDetails userDetails) {
        final String username = getUsernameFromToken(token);
        return (username.equals(userDetails.getUsername()) && !isTokenExpired(token));
    }
    
    /**
     * 从Token获取用户名
     */
    public String getUsernameFromToken(String token) {
        return getClaimFromToken(token, Claims::getSubject);
    }
    
    /**
     * 检查Token是否过期
     */
    private Boolean isTokenExpired(String token) {
        final Date expiration = getExpirationDateFromToken(token);
        return expiration.before(new Date());
    }
}
```

### 4.2 Security配置

```java
@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig {
    
    @Autowired
    private JwtAuthenticationEntryPoint jwtAuthenticationEntryPoint;
    
    @Autowired
    private UserDetailsService userDetailsService;
    
    @Bean
    public JwtAuthenticationFilter jwtAuthenticationFilter() {
        return new JwtAuthenticationFilter();
    }
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
    
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.cors().and().csrf().disable()
                .exceptionHandling().authenticationEntryPoint(jwtAuthenticationEntryPoint).and()
                .sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS).and()
                .authorizeHttpRequests(authz -> authz
                        .requestMatchers("/api/auth/**").permitAll()
                        .requestMatchers("/api/common/files/download/**").permitAll()
                        .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()
                        .requestMatchers("/api/teacher/**").hasRole("TEACHER")
                        .requestMatchers("/api/student/**").hasRole("STUDENT")
                        .anyRequest().authenticated()
                );
        
        http.addFilterBefore(jwtAuthenticationFilter(), UsernamePasswordAuthenticationFilter.class);
        
        return http.build();
    }
}
```

## 五、Redis缓存设计

### 5.1 Redis配置

```java
@Configuration
@EnableCaching
public class RedisConfig {
    
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        
        // 使用Jackson2JsonRedisSerializer来序列化和反序列化redis的value值
        Jackson2JsonRedisSerializer<Object> serializer = new Jackson2JsonRedisSerializer<>(Object.class);
        ObjectMapper mapper = new ObjectMapper();
        mapper.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        mapper.activateDefaultTyping(LazyLoadingEnabled.LAZY_LOADING_ENABLED, ObjectMapper.DefaultTyping.NON_FINAL);
        serializer.setObjectMapper(mapper);
        
        template.setValueSerializer(serializer);
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setHashValueSerializer(serializer);
        template.afterPropertiesSet();
        
        return template;
    }
    
    @Bean
    public CacheManager cacheManager(RedisConnectionFactory connectionFactory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofMinutes(30))
                .serializeKeysWith(RedisSerializationContext.SerializationPair.fromSerializer(new StringRedisSerializer()))
                .serializeValuesWith(RedisSerializationContext.SerializationPair.fromSerializer(new Jackson2JsonRedisSerializer<>(Object.class)));
        
        return RedisCacheManager.builder(connectionFactory)
                .cacheDefaults(config)
                .build();
    }
}
```

### 5.2 Redis工具类

```java
@Component
public class RedisUtils {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    /**
     * 设置缓存
     */
    public void set(String key, Object value, long timeout, TimeUnit unit) {
        redisTemplate.opsForValue().set(key, value, timeout, unit);
    }
    
    /**
     * 获取缓存
     */
    public Object get(String key) {
        return redisTemplate.opsForValue().get(key);
    }
    
    /**
     * 删除缓存
     */
    public Boolean delete(String key) {
        return redisTemplate.delete(key);
    }
    
    /**
     * 批量删除缓存
     */
    public Long delete(Collection<String> keys) {
        return redisTemplate.delete(keys);
    }
    
    /**
     * 设置过期时间
     */
    public Boolean expire(String key, long timeout, TimeUnit unit) {
        return redisTemplate.expire(key, timeout, unit);
    }
    
    /**
     * 获取过期时间
     */
    public Long getExpire(String key) {
        return redisTemplate.getExpire(key);
    }
    
    /**
     * 判断key是否存在
     */
    public Boolean hasKey(String key) {
        return redisTemplate.hasKey(key);
    }
}
```

## 六、文件管理

### 6.1 文件上传接口

```java
@RestController
@RequestMapping("/api/common/files")
@Api(tags = "文件管理")
public class FileController {
    
    @Autowired
    private FileService fileService;
    
    /**
     * 文件上传
     */
    @PostMapping("/upload")
    @ApiOperation("文件上传")
    public Result<FileUploadResponse> uploadFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam(required = false) String category) {
        
        FileUploadResponse response = fileService.uploadFile(file, category);
        return Result.success(response);
    }
    
    /**
     * 批量文件上传
     */
    @PostMapping("/batch-upload")
    @ApiOperation("批量文件上传")
    public Result<List<FileUploadResponse>> batchUploadFiles(
            @RequestParam("files") MultipartFile[] files,
            @RequestParam(required = false) String category) {
        
        List<FileUploadResponse> response = fileService.batchUploadFiles(files, category);
        return Result.success(response);
    }
    
    /**
     * 文件下载
     */
    @GetMapping("/download/{fileId}")
    @ApiOperation("文件下载")
    public void downloadFile(@PathVariable String fileId, HttpServletResponse response) {
        fileService.downloadFile(fileId, response);
    }
    
    /**
     * 文件在线预览
     */
    @GetMapping("/preview/{fileId}")
    @ApiOperation("文件在线预览")
    public Result<FilePreviewResponse> previewFile(@PathVariable String fileId) {
        FilePreviewResponse response = fileService.previewFile(fileId);
        return Result.success(response);
    }
    
    /**
     * 删除文件
     */
    @DeleteMapping("/{fileId}")
    @ApiOperation("删除文件")
    public Result<Void> deleteFile(@PathVariable String fileId) {
        fileService.deleteFile(fileId);
        return Result.success();
    }
}
```

## 七、AI功能预留接口

### 7.1 AI服务接口

```java
@RestController
@RequestMapping("/api/ai")
@Api(tags = "AI功能")
public class AIController {
    
    @Autowired
    private AIService aiService;
    
    /**
     * 智能内容推荐
     */
    @PostMapping("/recommend")
    @ApiOperation("智能内容推荐")
    public Result<RecommendationResponse> getRecommendations(
            @RequestBody @Valid RecommendationRequest request) {
        // TODO: 接入AI推荐算法
        RecommendationResponse response = aiService.getRecommendations(request);
        return Result.success(response);
    }
    
    /**
     * 智能批改
     */
    @PostMapping("/auto-grade")
    @ApiOperation("智能批改")
    public Result<AutoGradeResponse> autoGrade(
            @RequestBody @Valid AutoGradeRequest request) {
        // TODO: 接入AI批改服务
        AutoGradeResponse response = aiService.autoGrade(request);
        return Result.success(response);
    }
    
    /**
     * 知识图谱生成
     */
    @PostMapping("/knowledge-graph")
    @ApiOperation("知识图谱生成")
    public Result<KnowledgeGraphResponse> generateKnowledgeGraph(
            @RequestBody @Valid KnowledgeGraphRequest request) {
        // TODO: 接入知识图谱生成服务
        KnowledgeGraphResponse response = aiService.generateKnowledgeGraph(request);
        return Result.success(response);
    }
    
    /**
     * 学习能力分析
     */
    @PostMapping("/ability-analysis")
    @ApiOperation("学习能力分析")
    public Result<AbilityAnalysisResponse> analyzeAbility(
            @RequestBody @Valid AbilityAnalysisRequest request) {
        // TODO: 接入学习能力分析服务
        AbilityAnalysisResponse response = aiService.analyzeAbility(request);
        return Result.success(response);
    }
    
    /**
     * 智能问答
     */
    @PostMapping("/qa")
    @ApiOperation("智能问答")
    public Result<QAResponse> intelligentQA(
            @RequestBody @Valid QARequest request) {
        // TODO: 接入智能问答服务
        QAResponse response = aiService.intelligentQA(request);
        return Result.success(response);
    }
}
```

## 八、异常处理

### 8.1 全局异常处理器

```java
@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {
    
    /**
     * 业务异常
     */
    @ExceptionHandler(BusinessException.class)
    public Result<Void> handleBusinessException(BusinessException e) {
        log.error("业务异常: {}", e.getMessage());
        return Result.error(e.getCode(), e.getMessage());
    }
    
    /**
     * 参数校验异常
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<Void> handleValidationException(MethodArgumentNotValidException e) {
        BindingResult bindingResult = e.getBindingResult();
        String message = bindingResult.getFieldErrors().stream()
                .map(FieldError::getDefaultMessage)
                .collect(Collectors.joining(", "));
        log.error("参数校验异常: {}", message);
        return Result.error(400, message);
    }
    
    /**
     * 权限异常
     */
    @ExceptionHandler(AccessDeniedException.class)
    public Result<Void> handleAccessDeniedException(AccessDeniedException e) {
        log.error("权限异常: {}", e.getMessage());
        return Result.error(403, "权限不足");
    }
    
    /**
     * 系统异常
     */
    @ExceptionHandler(Exception.class)
    public Result<Void> handleException(Exception e) {
        log.error("系统异常: ", e);
        return Result.error("系统异常，请联系管理员");
    }
}
```

## 九、性能优化建议

### 9.1 数据库优化

- 使用MyBatis-Plus的分页插件
- 合理使用索引
- 避免N+1查询问题
- 使用批量操作

### 9.2 缓存策略

- 热点数据缓存（用户信息、课程列表等）
- 查询结果缓存（成绩统计、班级信息等）
- 分布式锁防止缓存击穿

### 9.3 异步处理

```java
@Service
public class AsyncTaskService {
    
    @Async
    public CompletableFuture<Void> processLargeDataAsync(List<Data> dataList) {
        // 异步处理大量数据
        return CompletableFuture.completedFuture(null);
    }
    
    @Async
    public void sendEmailAsync(String to, String subject, String content) {
        // 异步发送邮件
    }
}
```

这个后端架构设计提供了完整的RESTful API接口，支持教师和学生的所有核心功能，并为AI功能扩展预留了充足的接口和扩展点。所有接口都遵循统一的设计规范，便于团队开发和维护。