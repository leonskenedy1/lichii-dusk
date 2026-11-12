# نسخهٔ تشخیصی Lichii-Dusk — فقط ثبت اطلاعات (بدون تغییر منطق رندر)

## مهم: مبنای سورس و وضعیت اعتبارسنجی

این بسته بر اساس ZIP سورس `lichii-dusk-duskbloods-v2.zip` که در همین گفتگو ارائه شده بود ساخته شده است. آن ZIP فاقد پوشهٔ `.git` و شناسهٔ SCM قابل‌اتکا بود؛ بنابراین نمی‌توان تأیید کرد که دقیقاً همان درخت منبع Revision مربوط به `Lichii build-4` (`9878157d50c93ebf28370db5b3c08f18f2d34544`) است.

دو اصلاح RenderDoc در این بسته حفظ شده‌اند:

1. ثبت ابزارهای گرافیکی متصل، از جمله RenderDoc، و مسیر پیش‌فرض خروجی RenderDoc در لاگ Vulkan؛ این کد در ZIP ورودی وجود داشت و دست‌نخورده باقی مانده است.
2. محافظت از فراخوانی `GetCooperativeMatrixPropertiesKHR()`؛ این تابع اکنون فقط وقتی پرس‌وجو می‌شود که `VK_KHR_cooperative_matrix` در فهرست Extensionهای فعال/قابل‌استفاده باشد. این محافظ در ZIP ورودی موجود نبود و در این نسخه دوباره اعمال شده است.

همچنین کد موجود مربوط به رفتار تخصیص حافظه هنگام اتصال ابزار Debug حفظ شده است.

**این بسته در محیط فعلی کامپایل نشده است**؛ وابستگی‌ها و هدرهای Vulkan/سایر کتابخانه‌های موردنیاز در محیط ساخت این بسته حاضر نبودند. بررسی ایستای تعادل براکت‌ها، فاصله‌های انتهایی و تعداد آرگومان/placeholder برای ۲۴۹ فراخوانی Trace بدون خطا انجام شده است؛ این بررسی جایگزین Build واقعی نیست.

## دقیقاً چه چیزی اضافه شده است؟

ثبت تشخیصی جدید فقط با متغیر محیطی زیر فعال می‌شود:

```powershell
$env:EDEN_NX2_VULKAN_TRACE = '1'
```

خروجی جداگانه در این فایل نوشته می‌شود:

```text
%APPDATA%\eden\log\eden_gpu_trace.log
```

اگر فایل قبلی وجود داشته باشد، به `eden_gpu_trace.log.old.txt` تغییر نام داده می‌شود.

Trace شامل زمان monotonic میکروثانیه‌ای، شمارهٔ ترتیبی، و شناسهٔ Thread است. نوشتن فایل بین Threadها سریال شده است. فایل هر ۱ MiB Flush می‌شود و در Shutdown نیز Flush می‌شود. خطاهای خود سیستم Trace گرفته می‌شوند تا از مسیرهای `noexcept` به بیرون پرتاب نشوند.

گروه‌های اصلی اطلاعات:

- مرز فریم‌های Renderer و شمارندهٔ فریم میزبان.
- ثبت callbackهای Vulkan Validation پیش از فیلترهای قدیمی؛ شامل Severity، Message ID، متن کامل، نام/Handle آبجکت‌ها و Queue/Command Buffer labels.
- ساخت و نابودی Device/Queue/Swapchain و اشیای Vulkan؛ اطلاعات Pipeline، Shader Module، Shader Stage، Render Pass، Subpass، Attachment و Framebuffer.
- ثبت Bind شدن Pipelineها، Shader Hashهای گرافیکی/Compute، Descriptor Setها، Descriptor Bufferها، Descriptor Write/Copyها و Samplerها.
- تمام Pipeline Barrierهای مسیر wrapper که از آن عبور می‌کنند: Stage Mask، Access Mask، Image Layout قدیم/جدید، Queue Family، Aspect، Mip و Array Layer.
- Eventهای Vulkan، Timeline Semaphore، Wait/Signal، Fence، Queue Submit/Submit2، Acquire/Present و Device Wait Idle.
- عملیات Copy/Blit/Resolve/Clear و جزئیات Regionهای Buffer/Image.
- ایجاد Image/Buffer و مشخصات Allocation، Format، Extent، Usage، Memory Type، Offset، Size و Coherency.
- Snapshot محدود از Blitهای Fermi2D که از قیود معمول Source Copy دوبعدی خارج‌اند؛ حداکثر ۲۵۶ رخداد از هر نوع برای جلوگیری از رشد نامحدود لاگ.

`eden_gpu.log` و قالب قدیمی GPU Logging تغییر نکرده‌اند. این Trace یک فایل جداگانه و اختیاری است.

## اجرای آزمایش

### پیش‌نیازهای تنظیمات Lichii

برای این نوبت، تنظیمات قبلی را حفظ کن و فقط Trace را اضافه کن:

- Graphics Debugging: روشن
- GPU Logging Level: `All`
- GPU Vulkan Call Logging: روشن (در صورتی که در این Build گزینه‌اش موجود است؛ لاگ قبلی نشان می‌داد این مسیر فعال بوده)
- Disable Buffer Reorder: همان وضعیت قبلی (روشن)
- GPU Accuracy: همان مقدار قبلی (`Low`)
- GPU Fence Behavior: همان مقدار قبلی (`Default`)
- Sync Memory Operations: تغییر نده؛ در Build تو گزینه پیدا نشده و نباید برای این آزمایش تنظیمی حدس زده شود.
- Async GPU Emulation / Async Shader Compilation / Async Presentation: همان مقادیر قبلی؛ تغییر نده.
- Lichii Anti-Aliasing: همان `None` قبلی.

در وضعیت بازی:

- فایل‌های اصلی Drawparam را بازیابی کن؛ پوشهٔ Drawparam حذف یا جایگزین‌شده نباشد.
- در Debug Server مطمئن شو `rejected` خالی است.
- هیچ شیدری را Reject نکن.
- RenderDoc را به این اجرای عادی وصل نکن؛ اصلاحات RenderDoc در سورس حفظ شده‌اند، اما این آزمایش به Capture آن نیاز ندارد.

### متغیر محیطی Trace

متغیرهای Validation که در آزمایش قبلی با موفقیت باعث بارگذاری `VK_LAYER_KHRONOS_validation` و گزارش Synchronization Validation شدند را دقیقاً حفظ کن. فقط متغیر زیر را نیز به همان PowerShell یا فایل BAT اضافه کن:

```powershell
$env:EDEN_NX2_VULKAN_TRACE = '1'
```

اگر بازی را از PowerShell اجرا می‌کنی، بعد از تنظیم متغیرهای Validation و Trace، همان `lichii.exe` را اجرا کن. متغیرها باید پیش از شروع فرایند تنظیم شده باشند.

### مدت و صحنهٔ آزمایش

1. بازی را در همان صحنه/موقعیتی اجرا کن که فلیکر به‌وضوح دیده می‌شود.
2. تا حد امکان دوربین و موقعیت را ثابت نگه دار.
3. ۲۰ تا ۳۰ ثانیه از زمان مشاهدهٔ فلیکر کافی است؛ برای جلوگیری از تولید لاگ بسیار بزرگ، فعلاً اجرای چنددقیقه‌ای نداشته باش.
4. در صورت امکان، یک ویدئوی ۱۵ تا ۳۰ ثانیه‌ای از همان صحنه هم ضبط کن. این ویدئو برای فلیکر زمانی مهم است.
5. بازی را عادی ببند تا فایل Trace و GPU Log Flush شوند.

## فایل‌هایی که باید ارسال شوند

از `%APPDATA%\eden\log\` این موارد را بفرست:

1. `eden_log.txt`
2. `eden_gpu.log`
3. `eden_gpu_trace.log`
4. ویدئوی کوتاه از صحنهٔ دارای فلیکر، در صورت امکان.

اگر `eden_gpu_trace.log` ایجاد نشد، در `eden_log.txt` دنبال پیام زیر بگرد:

```text
Detailed Vulkan trace enabled
```

اگر پیام وجود ندارد، بررسی می‌کنیم که متغیر محیطی به همان فرایند Lichii منتقل شده و GPU Logging روی `Off` نبوده باشد.

اگر Trace بسیار بزرگ شد، سه فایل لاگ را با هم ZIP/RAR کن؛ فایل‌ها را ویرایش یا کوتاه نکن.

## محدودهٔ تغییرات رفتاری

هیچ Stage/Access Mask، Barrier، Layout، Semaphore، Wait، Queue Submit، Present، انتخاب Pipeline، Shader، Format، مقدار AA یا تنظیم Renderer به قصد رفع مشکل تغییر داده نشده است. فراخوانی‌های Vulkan اصلی همان تعداد و همان ترتیب قبلی را حفظ می‌کنند؛ فقط قبل/بعد آن‌ها دادهٔ تشخیصی ثبت می‌شود.

در چند نقطه برای ثبت نتیجهٔ فراخوانی، مقدار برگشتی قبلی در متغیر محلی ذخیره شده و همان مقدار قبلی به مسیر اصلی بازگردانده می‌شود. در مسیر ساخت Compute Pipeline، Handle اولیه به `VK_NULL_HANDLE` مقداردهی شده تا در صورت شکست ایجاد Pipeline، مقدار نامعین برای ثبت لاگ خوانده نشود؛ مسیر بررسی نتیجه و موفقیت/شکست تغییری نکرده است.
